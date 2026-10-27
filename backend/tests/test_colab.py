from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import pytest

import benchmark
import colab_runner
from app.evaluation.colab_models import ColabModels, EMBEDDING_ID, token_windows, strip_json_fence
from app.services import qwen_client, retrieval


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()
    def decode(self, tokens, **kwargs):
        return " ".join(tokens)


def test_windowing_covers_long_sources_and_the_final_tokens():
    text = " ".join(str(i) for i in range(1000))
    windows = token_windows(Tokenizer(), text, 240, 32)
    assert set(" ".join(windows).split()) == set(text.split())
    assert windows[-1].endswith("999")
    assert all(len(window.split()) <= 240 for window in windows)


def test_windowing_rejects_a_nonadvancing_stride():
    with pytest.raises(ValueError):
        token_windows(Tokenizer(), "some text", 10, 10)


def test_json_fences_are_removed_without_inventing_missing_fields():
    assert strip_json_fence('```json\n{"verdicts": []}\n```') == '{"verdicts": []}'
    assert strip_json_fence('nonsense') == 'nonsense'


def test_open_weight_adapter_blocks_hosted_api_and_restores_transports():
    models = ColabModels.__new__(ColabModels)
    models.revisions = {EMBEDDING_ID: "test-revision"}
    models.embed = lambda texts: np.ones((len(texts), 2))
    models.rerank = lambda query, candidates: candidates[:4]
    models.chat = lambda **kwargs: None
    original = qwen_client.get_client
    with models.activate():
        assert benchmark.EMBEDDING_MODEL.startswith(EMBEDDING_ID)
        assert benchmark.embed_texts(["text"]).shape == (1, 2)
        with pytest.raises(RuntimeError, match="disabled"):
            qwen_client.get_api_key()
        with pytest.raises(RuntimeError, match="disabled"):
            retrieval._embedding_client()
        with pytest.raises(RuntimeError, match="disabled"):
            retrieval._rerank_client()
    assert qwen_client.get_client is original


def test_checkpoint_resume_does_not_repeat_completed_chunks(monkeypatch, tmp_path):
    dataset_path = Path(__file__).resolve().parents[2] / "evaluation/datasets/scifact-v2/development.json"
    calls = []
    def run(path, *args, **kwargs):
        assert kwargs["verdict_batch_size"] == 1
        cases = json.loads(path.read_text())["cases"]
        calls.append([c["id"] for c in cases])
        rows = [{"case_id": c["id"], "claim_id": item["id"],
                 "expected_verdict": item["expected_verdict"],
                 "actual_verdict": None, "expected_source_ids": item["expected_source_ids"],
                 "ranked_source_ids": []}
                for c in cases for item in c["claims"]]
        return {"evaluated_cases": len(cases), "strategies": {
            name: {"claim_results": rows, "errors": [], "reranker_fallback_claims": 0}
            for name in ["cosine", "rerank"]}}
    monkeypatch.setattr(colab_runner.benchmark, "run", run)
    class Models:
        metadata = {"profile": "mock-not-model-quality"}
        activate = lambda self: nullcontext()
    first = colab_runner.run_resumable(dataset_path, Models(), tmp_path, limit=2, chunk_size=2)
    assert first["completed_cases"] == 2
    assert first["partial_run"]
    second = colab_runner.run_resumable(dataset_path, Models(), tmp_path, limit=2, chunk_size=2)
    assert len(calls) == 1
    assert second["strategies"]["cosine"]["metrics"]["missing_verdicts"] == 2
    third = colab_runner.run_resumable(dataset_path, Models(), tmp_path, limit=4, chunk_size=2)
    assert len(calls) == 2
    assert third["completed_cases"] == 4
    assert len({r["claim_id"] for r in third["strategies"]["cosine"]["claim_results"]}) == 4


def test_checkpoint_runner_rejects_unknown_mode_before_work(tmp_path):
    class Models:
        metadata = {}
    with pytest.raises(ValueError, match="Choose"):
        colab_runner.run_resumable(tmp_path / "missing.json", Models(), tmp_path, mode="unknown")
