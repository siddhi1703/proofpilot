"""Import upstream expert annotations without relabeling them with an LLM."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import tarfile

from app.evaluation.lexical import BM25

ARCHIVE_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
SOURCE_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
LABELS = {"SUPPORT": "supported", "CONTRADICT": "contradicted"}


def read_archive(path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("SciFact archive differs from the pinned release checksum.")
    # Read only known regular members; never extract archive paths to disk.
    with tarfile.open(path) as archive:
        def read(name):
            member = archive.getmember("data/" + name)
            if not member.isfile() or member.size > 20_000_000:
                raise ValueError("Unexpected dataset member.")
            return [json.loads(line) for line in archive.extractfile(member)]
        return read("corpus.jsonl"), read("claims_train.jsonl"), read("claims_dev.jsonl")


def referenced_docs(claim: dict) -> set[int]:
    return set(claim["cited_doc_ids"]) | {int(key) for key in claim["evidence"]}


def build(archive_path: Path, output: Path, development_size: int = 100, pool_size: int = 16) -> dict:
    if pool_size <= 8:
        raise ValueError("Candidate pools must exceed the production top-eight limit.")
    corpus, train, validation = read_archive(archive_path)
    by_id = {doc["doc_id"]: doc for doc in corpus}
    test_gold_docs = set().union(*(referenced_docs(claim) for claim in validation))
    eligible_dev = [claim for claim in train if not referenced_docs(claim) & test_gold_docs]
    eligible_dev.sort(key=lambda claim: hashlib.sha256(f"proofpilot-v2/{claim['id']}".encode()).hexdigest())
    development = eligible_dev[:development_size]
    if len(development) != development_size:
        raise ValueError("Not enough document-disjoint development claims.")
    development_gold_docs = set().union(*(referenced_docs(claim) for claim in development))
    assert not development_gold_docs & test_gold_docs
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split, claims, forbidden in [
        ("development", development, test_gold_docs),
        ("test", validation, development_gold_docs),
    ]:
        # Partition even distractors: no source document appears in both splits.
        # All test gold docs belong to test. Unassigned docs are hash-partitioned.
        docs = [doc for doc in corpus if doc["doc_id"] not in forbidden and (
            (doc["doc_id"] in development_gold_docs if split == "development" else doc["doc_id"] in test_gold_docs)
            or (doc["doc_id"] not in development_gold_docs | test_gold_docs and
                (int(hashlib.sha256(str(doc["doc_id"]).encode()).hexdigest(), 16) % 4 == 0) == (split == "development"))
        )]
        texts = [doc["title"] + ". " + " ".join(doc["abstract"]) for doc in docs]
        index = BM25(texts)
        cases = []
        for claim in claims:
            labels = {item["label"] for evidence in claim["evidence"].values() for item in evidence}
            if len(labels) > 1:
                # Do not invent a policy label for conflicting upstream annotations.
                raise ValueError(f"Mixed upstream labels for claim {claim['id']}")
            label = LABELS[next(iter(labels))] if labels else "unsupported"
            gold_ids = {int(key) for key in claim["evidence"]}
            selected_ids = list(sorted(gold_ids))
            for idx in index.rank(claim["claim"]):
                doc_id = docs[idx]["doc_id"]
                if doc_id not in selected_ids:
                    selected_ids.append(doc_id)
                if len(selected_ids) >= pool_size:
                    break
            random.Random(f"proofpilot-v2/{split}/{claim['id']}").shuffle(selected_ids)
            passages = [{
                "id": f"D{doc_id}", "page": 1,
                "filename": f"scifact-{doc_id}.txt",
                "text": by_id[doc_id]["title"] + ". " + " ".join(by_id[doc_id]["abstract"]),
            } for doc_id in selected_ids]
            cases.append({
                "id": f"scifact-{claim['id']}",
                "upstream_claim_id": claim["id"],
                "passages": passages,
                "claims": [{
                    "id": f"C{claim['id']}", "text": claim["claim"],
                    "type": "general_fact", "importance": "high",
                    "expected_verdict": label,
                    "expected_source_ids": [f"D{doc_id}" for doc_id in sorted(gold_ids)],
                    "evidence_groups": [
                        {"source_id": f"D{doc_id}",
                         "sentences": [by_id[int(doc_id)]["abstract"][i] for i in item["sentences"]]}
                        for doc_id, evidence in claim["evidence"].items() for item in evidence
                    ],
                }],
            })
        data = {
            "schema_version": 3, "dataset_id": "scifact-proofpilot-v2",
            "split": split, "annotation_status": "upstream expert annotations; label mapping only",
            "source_url": SOURCE_URL, "archive_sha256": ARCHIVE_SHA256,
            "upstream_split": "train" if split == "development" else "dev",
            "candidate_pool_size": pool_size,
            "protocol": "gold evidence abstracts plus lexical hard negatives; not full-corpus retrieval",
            "label_mapping": {**LABELS, "NO_EVIDENCE": "unsupported"},
            "limitations": [
                "Public benchmark may be present in model pretraining.",
                "Test is upstream development, not the unreleased SciFact test set.",
                "Unsupported means no upstream annotated evidence, not proof of falsehood.",
                "Gold evidence is included in the candidate pool by construction.",
                "Abstracts are passages; page number 1 is a container marker, not a journal page.",
                "Only three verdict classes are expert annotated; other policy classes use separate tests.",
            ],
            "cases": cases,
        }
        content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        (output / f"{split}.json").write_text(content, encoding="utf-8")
        summaries[split] = {
            "claims": len(cases), "candidate_pool_size": pool_size,
            "labels": dict(Counter(case["claims"][0]["expected_verdict"] for case in cases)),
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
    # Verify source and claim separation using the written, frozen datasets.
    datasets = {s: json.loads((output / f"{s}.json").read_text()) for s in summaries}
    source_sets = {s: {p["id"] for c in d["cases"] for p in c["passages"]} for s, d in datasets.items()}
    claim_sets = {s: {c["id"] for c in d["cases"]} for s, d in datasets.items()}
    if source_sets["development"] & source_sets["test"] or claim_sets["development"] & claim_sets["test"]:
        raise ValueError("Development/test leakage detected.")
    manifest = {"dataset_id": "scifact-proofpilot-v2", "source": SOURCE_URL,
                "archive_sha256": ARCHIVE_SHA256, "splits": summaries,
                "claim_overlap": 0, "source_document_overlap": 0}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
