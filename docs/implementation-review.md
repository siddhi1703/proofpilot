# ProofPilot implementation review - October 27, 2026

## Delivered locally

- Production retrieval now supports a genuine cosine-only baseline and the
  reranking policy using the same embedding inputs. It exposes per-claim
  reranker failures, validates embedding counts/shapes/finite values, rejects
  duplicate source IDs, and uses stable ranking for tied similarities.
- Verdict validation no longer repairs model paraphrases into approved quotes.
  Missing/duplicate verdicts, unknown citations, ungrounded quotes, and missing
  numeric coverage on supported claims require human review. Unknown claim IDs
  reject the model batch. These checks do not replace semantic judgment.
- No extracted claims means blocked/unverified. Every non-supported verdict
  blocks publication. Hitting the 12-claim extraction cap asks the user to split
  the draft, avoiding a potentially truncated report.
- Known provider quota, authentication, rate-limit, and outage responses map to
  safe actionable service errors rather than echoing raw provider responses.
- Audit notices appear on screen and in the existing PDF export. Existing UI
  work was preserved. PDF-text sanitation and history loading were adjusted to
  pass lint, and unavailable browser storage no longer discards a successful audit.

## Evaluation delivered

The official SciFact archive is pinned by SHA-256. Upstream expert labels are
retained with explicit mapping, provenance, attribution, and licensing.

- Development: 100 claims, selected from upstream train.
- Project test: all 300 upstream development claims, not the unreleased official test.
- Candidate pool: 16 abstracts per case, including known evidence plus lexical hard negatives.
- Verified zero shared claims or candidate source documents between our splits.
- Separate ten-example synthetic policy set covers all five application labels.
- Comparative runner supports BM25, cosine-only, and production reranking;
  embedding caching ensures neural strategies use the same vectors.
- Reports include per-claim rankings and verdicts, evidence quotes/citations,
  per-class precision/recall/F1, confusion matrices, accuracy intervals, missing
  outputs, false-supported counts, timings, fallbacks, and code/data/model hashes.

The full 300-case offline BM25 run produced Recall@4 **0.8183**, averaged over
188 evidence-bearing claims. Hit rate@4 was **0.8351**. The remaining 112 claims
have no labeled evidence and are excluded from retrieval denominators. These
are lexical retrieval results, not LLM verdict accuracy or full-corpus scores.
See `evaluation/reports/scifact-v2-test-bm25.json` for inspectable results and
`evaluation/README.md` for exact reproduction commands and limitations.

## Verification

- Backend: 61 tests passed locally, including API integration, safety behavior,
  malformed embeddings, fallback visibility, missing-output scoring, dataset
  separation, and the complete comparison runner with mocked provider outputs.
- Frontend: production build and ESLint passed.
- Whitespace/diff check passed.
- Hosted embedding preflight reached Qwen but returned HTTP 400,
  `AllocationQuota.FreeTierOnly`: the account's free quota was exhausted.
- A free Google Colab Tesla T4 successfully ran the bundled tests and a ten-case
  development pilot using MiniLM embeddings, a MiniLM cross-encoder, and
  NF4 four-bit Qwen3-4B-Instruct-2507. The report records resolved model revisions.
  Both retrieval strategies yielded 8/10 correct verdicts, with no missing outputs,
  infrastructure errors, or reranker fallbacks. This subset contains six supported
  and four unsupported claims, with no contradictions; the 95% accuracy interval
  is approximately 49.0%-94.3%. It is a runtime check, not a reliable quality claim.
- The complete 100-case development run achieved 50% exact-label accuracy,
  macro F1 0.5842, with no missing predictions, infrastructure errors, reranker
  fallbacks, or observed false-supported predictions. Support recall was 36.17%;
  the model often chose weak support or human review. These limitations remain
  visible rather than being remapped into correct answers.
- Complete 300-case neural retrieval comparison: cosine Recall@4 92.25%, reranked
  Recall@4 94.47%, across 188 evidence-bearing claims in 16-abstract pools.
  Reranked Recall@1 was 80.01%, versus cosine 69.33%.
- A separate three-class NLI diagnostic scored 65.67% on the frozen 300-case test,
  with 22 false-supported outputs. It differs from the five-verdict application
  policy and is not an automatic-approval replacement. Its configuration was
  selected on development and frozen before test.
- The final five-verdict Qwen test report has not been recovered after the
  browser/session interruption. The last confirmed checkpoint was 40/300.
  Inference code and model configuration remain frozen; no final score is claimed.
- The portable notebook now restores legacy and manifest-backed checkpoint ZIPs,
  rejects unsafe paths and conflicting results before writing, and preserves
  existing retrieval-report bytes referenced by NLI provenance hashes.
- The live application demo correctly blocked an invented claim. Synthetic
  policy-v3 examples scored 10/10 after matching production claim-ID format;
  the original failed PC-prefixed run is preserved and explained.
- Drive mounting failed with a credential-propagation error. The session uses
  Colab's native file upload and checkpoint ZIPs; the pilot report and archive
  were backed up to the user's selected Google Drive account.

## Continuing evaluation

Use the free [Colab workflow](../colab/README.md) while hosted quota is unavailable.
The paid-provider command below is an alternative only if access is configured.

From `backend/`, with its virtual environment active:

```bash
python benchmark.py run \
  --dataset ../evaluation/datasets/scifact-v2/development.json \
  --mode compare --verdicts \
  --output ../evaluation/reports/scifact-v2-development-qwen.json
```

Review development failures before a fixed test run. Freeze prompts/code before
running `test.json`; do not tune to test scores. Record a separate policy run.
A versioned newsroom-domain evaluation and larger-corpus retrieval are still
needed for claims beyond this controlled scientific-abstract benchmark.

Changes are local and uncommitted. Nothing has been pushed or deployed, and the
historical 12-claim results have been preserved.
