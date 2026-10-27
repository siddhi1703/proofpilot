# ProofPilot: updated resume wording

Updated October 27, 2026. See `implementation-review.md` for the implementation,
verification, and live-model evaluation status.

## Recommended entry now

**ProofPilot - Evidence Audit Platform | Python, FastAPI, React, PyTorch, Qwen**

- Built a full-stack evidence-auditing platform using page-aware RAG, embeddings,
  and reranking to check draft claims against PDF/TXT sources; implemented exact
  citation checks, numeric coverage checks, privacy scanning, and human review.
- Benchmarked retrieval on 300 expert-labeled SciFact claims in controlled
  16-abstract pools; cross-encoder reranking improved Recall@1 from 69.33% to
  80.01% across the 188 claims with annotated evidence.

This describes implemented capabilities. It does not claim that live Qwen
quality has been validated on all 400 claims. Do not say "400 claims trained";
this dataset is for evaluation, not model training.

## Measured results

All 300 test cases have completed retrieval comparison. Across 188 claims with
annotated evidence, cosine Recall@4 was 92.25% and reranked Recall@4 was 94.47%.
The BM25 baseline reached 81.83%. The remaining 112 claims have no retrieval
target. These are controlled candidate-pool results, not full-corpus results or
fact-checking accuracy.

The original five-verdict Qwen Colab profile achieved 50% exact-label development
accuracy with zero observed false-supported predictions, but low support recall.
Its final 300-case test report has not been recovered. A separate three-class NLI diagnostic completed the
test at 65.67% accuracy, with 22 false-supported predictions. Its policy differs
from the application's five verdicts. Keep these limitations in the project
README; neither result supports an autonomous fact-checking claim.
Keep the old 12-claim perfect scores out of the headline resume entry.

## Remaining portfolio sequence

Next review the meeting-notes project for data separation, baseline
comparability, ROUGE precision/F1 as well as recall, and speaker-attributed
outputs. Then review crater detection for its exact metric and evaluation split,
and verify publication and competition evidence.
