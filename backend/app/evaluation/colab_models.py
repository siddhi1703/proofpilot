"""Open-weight inference on a Colab GPU; no hosted inference API calls."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

EMBEDDING_ID = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
VERDICT_ID = "Qwen/Qwen3-4B-Instruct-2507"


def token_windows(tokenizer, text: str, size: int, overlap: int) -> list[str]:
    """Cover all tokens rather than silently discarding long abstracts."""
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("Invalid token window configuration.")
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return [text]
    windows = []
    for start in range(0, len(ids), size - overlap):
        windows.append(tokenizer.decode(ids[start:start + size], skip_special_tokens=True))
        if start + size >= len(ids):
            break
    return windows


def strip_json_fence(text: str) -> str:
    """Permit a single Markdown fence; do not repair or invent JSON values."""
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        return text[8:-3].strip()
    if text.startswith("```\n") and text.endswith("```"):
        return text[4:-3].strip()
    return text


class ColabModels:
    def __init__(self):
        import torch
        from huggingface_hub import model_info
        if not torch.cuda.is_available():
            raise RuntimeError("Select Runtime > Change runtime type > T4 GPU. No paid API fallback will be used.")
        if torch.cuda.get_device_properties(0).total_memory < 12 * 1024 ** 3:
            raise RuntimeError("This profile needs a GPU with at least 12 GiB VRAM.")
        self.torch = torch
        self.revisions = {name: model_info(name).sha for name in [EMBEDDING_ID, RERANK_ID, VERDICT_ID]}
        self.encoder = self.reranker = self.rerank_tokenizer = self.llm = self.tokenizer = None
        self.metadata = {
            "backend": "colab-open-weights", "revisions": self.revisions,
            "embedding": EMBEDDING_ID, "rerank": RERANK_ID, "verdict": VERDICT_ID,
            "embedding_policy": "240-token windows, 32-token overlap, mean of normalized window vectors then L2 normalization",
            "rerank_policy": "maximum cross-encoder logit over 320-token source windows with 64-token overlap; no lexical rescue",
            "quantization": "verdict model NF4 4-bit, float16 compute, double quantization",
            "generation": "greedy; one claim per verdict request; 2048 output-token cap; 12000 input-token cap; no silent truncation",
            "gpu": torch.cuda.get_device_name(0),
        }

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(EMBEDDING_ID, revision=self.revisions[EMBEDDING_ID], device="cuda")
        windows, spans = [], []
        for text in texts:
            start = len(windows)
            windows.extend(token_windows(self.encoder.tokenizer, text, 240, 32))
            spans.append((start, len(windows)))
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        vectors = self.encoder.encode(windows, batch_size=16, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)
        pooled = np.vstack([vectors[start:end].mean(axis=0) for start, end in spans]).astype(np.float32)
        return pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)

    def rerank(self, query, candidates):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        torch = self.torch
        if self.reranker is None:
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(RERANK_ID, revision=self.revisions[RERANK_ID])
            self.reranker = AutoModelForSequenceClassification.from_pretrained(
                RERANK_ID, revision=self.revisions[RERANK_ID], torch_dtype=torch.float16,
            ).to("cuda").eval()
        if len(self.rerank_tokenizer.encode(query, add_special_tokens=False)) > 160:
            raise ValueError("Claim is too long for this reranker profile; it was not truncated.")
        pairs, owners = [], []
        for i, candidate in enumerate(candidates):
            for window in token_windows(self.rerank_tokenizer, candidate.text, 320, 64):
                pairs.append((query, window))
                owners.append(i)
        scores = np.full(len(candidates), -np.inf)
        for start in range(0, len(pairs), 8):
            batch = pairs[start:start + 8]
            inputs = self.rerank_tokenizer([q for q, _ in batch], [p for _, p in batch],
                                          padding=True, truncation=False, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                logits = self.reranker(**inputs).logits.float().flatten().cpu().numpy()
            if not np.isfinite(logits).all():
                raise ValueError("The reranker returned non-finite scores.")
            for owner, value in zip(owners[start:start + 8], logits, strict=True):
                scores[owner] = max(scores[owner], float(value))
        ranked = np.argsort(-scores, kind="stable")[:4]
        return [candidates[i].model_copy(update={"rerank_score": float(1 / (1 + np.exp(-np.clip(scores[i], -60, 60))))}) for i in ranked]

    def chat(self, *, messages, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        torch = self.torch
        if self.llm is None:
            self.tokenizer = AutoTokenizer.from_pretrained(VERDICT_ID, revision=self.revisions[VERDICT_ID])
            self.llm = AutoModelForCausalLM.from_pretrained(
                VERDICT_ID, revision=self.revisions[VERDICT_ID], device_map={"": 0},
                torch_dtype=torch.float16,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True),
                attn_implementation="sdpa",
            ).eval()
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        length = inputs["input_ids"].shape[1]
        if length > 12_000:
            raise ValueError("The evidence exceeds this Colab profile's context budget; it was not truncated.")
        with torch.inference_mode():
            generated = self.llm.generate(**inputs, do_sample=False, max_new_tokens=2048,
                pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(generated[0, length:], skip_special_tokens=True)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=strip_json_fence(text)))])

    @contextmanager
    def activate(self):
        """Reuse production verdict validation while replacing every paid transport."""
        import benchmark
        from app.services import qwen_client, retrieval
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.chat)))
        def forbidden(*args, **kwargs):
            raise RuntimeError("Hosted inference is disabled in the free Colab profile.")
        replacements = [
            (benchmark, "embed_texts", self.embed),
            (benchmark, "EMBEDDING_MODEL", EMBEDDING_ID + ":window-mean-v1:" + self.revisions[EMBEDDING_ID]),
            (benchmark, "RERANK_MODEL", RERANK_ID), (benchmark, "MODEL_NAME", VERDICT_ID),
            (retrieval, "embed_texts", self.embed), (retrieval, "_rerank", self.rerank),
            (retrieval, "EMBEDDING_MODEL", EMBEDDING_ID), (retrieval, "RERANK_MODEL", RERANK_ID),
            (retrieval, "_embedding_client", forbidden), (retrieval, "_rerank_client", forbidden),
            (qwen_client, "get_client", lambda: client), (qwen_client, "get_api_key", forbidden),
            (qwen_client, "MODEL_NAME", VERDICT_ID),
        ]
        saved = [(obj, name, getattr(obj, name)) for obj, name, value in replacements]
        try:
            for obj, name, value in replacements:
                setattr(obj, name, value)
            yield
        finally:
            for obj, name, value in reversed(saved):
                setattr(obj, name, value)
