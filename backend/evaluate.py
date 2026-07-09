import json
from pathlib import Path

from app.schemas.audit import ExtractedClaim
from app.services.document_parser import SourceChunk
from app.services.qwen_client import evaluate_claims
from app.services.retrieval import retrieve_evidence

BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "benchmark.json"


def evaluate() -> dict[str, float | int]:
    cases = json.loads(BENCHMARK_PATH.read_text())
    verdict_correct = 0
    recall_at_1 = 0
    recall_at_4 = 0
    total = 0

    for case in cases:
        chunks = [
            SourceChunk(
                id=passage["id"],
                text=passage["text"],
                page_number=passage["page"],
                filename=f"{case['id']}.txt",
            )
            for passage in case["passages"]
        ]
        claims = [
            ExtractedClaim(
                id=item["id"],
                text=item["text"],
                type=item["type"],
                importance=item["importance"],
            )
            for item in case["claims"]
        ]
        expected = {item["id"]: item for item in case["claims"]}

        retrieval = retrieve_evidence(claims, chunks)
        audits = evaluate_claims(claims, retrieval.evidence_by_claim)

        for audit in audits:
            expected_item = expected[audit.claim.id]
            ranked_source_ids = [
                passage.source_id
                for passage in retrieval.evidence_by_claim[audit.claim.id]
            ]
            total += 1
            verdict_correct += audit.verdict == expected_item["expected_verdict"]
            recall_at_1 += (
                bool(ranked_source_ids)
                and ranked_source_ids[0] == expected_item["expected_source_id"]
            )
            recall_at_4 += expected_item["expected_source_id"] in ranked_source_ids[:4]

            print(
                f"{case['id']}/{audit.claim.id}: "
                f"expected={expected_item['expected_verdict']} "
                f"actual={audit.verdict} "
                f"top={ranked_source_ids[:4]}"
            )

    return {
        "cases": len(cases),
        "claims": total,
        "verdict_accuracy": round(verdict_correct / total, 4),
        "retrieval_recall_at_1": round(recall_at_1 / total, 4),
        "retrieval_recall_at_4": round(recall_at_4 / total, 4),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
