# ProofPilot benchmark

This small, hand-authored benchmark checks whether ProofPilot retrieves the
known evidence passage and assigns the expected verdict. It contains three
domains, 18 passages, and 12 claims, including supported, unsupported, and
contradicted examples.

Run it from the backend virtual environment:

```bash
python evaluate.py
```

The script reports verdict accuracy and retrieval recall at 1 and 4. This is a
transparent hackathon benchmark, not a claim of production-scale validation.

## Recorded result

The June 29, 2026 run produced:

- Verdict accuracy: 12/12 (100%)
- Retrieval Recall@1: 6/12 (50%)
- Retrieval Recall@4: 12/12 (100%)

The result is stored in `results.json`. The benchmark is intentionally kept in
the repository so judges and reviewers can inspect every expected answer.
The recorded verdict model is `qwen3.6-flash`.
