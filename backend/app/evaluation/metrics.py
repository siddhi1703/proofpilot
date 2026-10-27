"""Metrics whose denominators include failed and missing predictions."""
from math import sqrt
from typing import get_args

import numpy as np

from app.schemas.audit import Verdict


def wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [round(max(0, center - radius), 4), round(min(1, center + radius), 4)]


def score(rows: list[dict], *, verdicts: bool) -> dict:
    count = len(rows)
    positive = [row for row in rows if row["expected_source_ids"]]
    results = {
        "claims": count, "retrieval_applicable_claims": len(positive),
        "retrieval_not_applicable_claims": count - len(positive),
    }
    for k in [1, 4, 8]:
        values = [len(set(row["ranked_source_ids"][:k]) & set(row["expected_source_ids"]))
                  / len(set(row["expected_source_ids"])) for row in positive]
        hits = [bool(set(row["ranked_source_ids"][:k]) & set(row["expected_source_ids"])) for row in positive]
        results[f"retrieval_recall_at_{k}"] = round(float(np.mean(values)), 4) if values else None
        results[f"retrieval_hit_rate_at_{k}"] = round(float(np.mean(hits)), 4) if hits else None
    reciprocal = []
    for row in positive:
        ranks = [i + 1 for i, sid in enumerate(row["ranked_source_ids"]) if sid in row["expected_source_ids"]]
        reciprocal.append(1 / min(ranks) if ranks else 0)
    results["retrieval_mrr_at_8"] = round(float(np.mean(reciprocal)), 4) if reciprocal else None
    if not verdicts:
        return results
    correct = sum(row.get("actual_verdict") == row["expected_verdict"] for row in rows)
    present = sum(row.get("actual_verdict") is not None for row in rows)
    per_label, confusion, f1s = {}, {}, []
    for label in get_args(Verdict):
        support = sum(row["expected_verdict"] == label for row in rows)
        predicted = sum(row.get("actual_verdict") == label for row in rows)
        tp = sum(row["expected_verdict"] == label and row.get("actual_verdict") == label for row in rows)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_label[label] = {"support": support, "predicted": predicted,
                            "precision": round(precision, 4),
                            "recall": round(recall, 4) if support else None,
                            "f1": round(f1, 4) if support else None}
        if support:
            f1s.append(f1)
        confusion[label] = {}
        for row in rows:
            if row["expected_verdict"] == label:
                pred = row.get("actual_verdict") or "missing"
                confusion[label][pred] = confusion[label].get(pred, 0) + 1
    supported = [row for row in rows if row.get("actual_verdict") == "supported"]
    results.update({
        "verdict_accuracy": round(correct / count, 4) if count else None,
        "accuracy_wilson_95": wilson(correct, count),
        "verdict_coverage": round(present / count, 4) if count else None,
        "missing_verdicts": count - present,
        "macro_f1_observed_labels": round(float(np.mean(f1s)), 4) if f1s else None,
        "per_verdict": per_label, "confusion_matrix": confusion,
        "false_supported": sum(row["expected_verdict"] != "supported" for row in supported),
    })
    return results
