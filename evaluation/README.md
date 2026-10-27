# Evaluation

ProofPilot now has a versioned public-data evaluation alongside the original
12-claim smoke test. Model scores must come from recorded runs, not unit tests.

## Datasets

| Dataset | Claims | Labels | Role |
| --- | ---: | --- | --- |
| SciFact development | 100 | 47 supported, 36 unsupported, 17 contradicted | Development and diagnosis |
| SciFact project test | 300 | 124 supported, 112 unsupported, 64 contradicted | Fixed evaluation after development |
| Policy examples v3 | 10 | Two per application verdict | Synthetic implementation examples |
| Historical smoke test | 12 | Three verdicts | Archived hackathon comparison only |

SciFact is expert annotated by the upstream authors, rather than generated or
labeled by this application. Our label mapping and candidate-pool adaptation
are documented in [the attribution](datasets/scifact-v2/ATTRIBUTION.md).
`manifest.json` records checksums, counts, and verified zero overlap of claims
and source documents between development and test, including distractors.

Each SciFact case includes its known evidence abstracts plus lexical hard
negatives, shuffled deterministically into a pool of 16 abstracts. Unsupported
claims have no labeled evidence. Each abstract is one passage. The first-stage
retriever must select eight candidates before the top four reach the verifier.
The gold abstracts are included by construction: this is controlled candidate-pool
retrieval, not full-corpus search. Sentence rationales are retained for inspection.

The benchmark receives the upstream claim text and abstract passages directly.
It measures retrieval and verdict behavior, not claim-extraction coverage, PDF
parsing accuracy, or privacy-detection quality. Those application paths have
separate regression tests and a small integration demo.

The two additional policy classes, weakly_supported and human_review_required,
are absent from SciFact. `datasets/policy-v3.json` supplies ten clearly marked
hand-authored examples. Do not combine their scores with the expert-data result
or present them as independently reviewed annotations.

## Recorded offline results

`reports/scifact-v2-test-bm25.json` contains the full 300-case offline BM25 run:

| Metric | Value | Denominator |
| --- | ---: | --- |
| Recall@1 | 0.6387 | 188 claims with labeled evidence |
| Recall@4 | 0.8183 | 188 claims with labeled evidence |
| Hit rate@4 | 0.8351 | 188 claims with labeled evidence |
| Recall@8 | 0.9029 | 188 claims with labeled evidence |
| MRR@8 | 0.7382 | 188 claims with labeled evidence |

Recall averages the fraction of labeled evidence abstracts recovered per claim.
Hit rate asks whether at least one was recovered. Claims with no annotated
evidence are excluded from retrieval denominators, but remain in verdict
accuracy denominators when verdict evaluation is requested.

These are lexical retrieval scores. No verdict accuracy, embedding quality,
reranking gain, or end-to-end latency is implied. The initial live embedding
preflight on October 26, 2026 returned `AllocationQuota.FreeTierOnly`: the
configured Qwen account had exhausted its free quota. No model evaluation
results were manufactured to replace that failed request.

The separate free Colab open-weight profile has completed a ten-case development
pilot, recorded in `reports/colab-development-pilot-10.json`. Both cosine and
reranked evidence produced 8/10 correct verdicts; the approximate 95% Wilson
interval is 0.4902-0.9433. The sample has six supported and four unsupported claims
and no contradicted claims. It establishes that the GPU workflow runs; it is too
small to establish model quality. Full development evaluation is recorded below.
See [Colab setup and model differences](../colab/README.md).

The full 100-case development retrieval comparison is recorded in
`reports/colab-development-retrieval-100.json` (64 evidence-bearing claims):

| Method | Recall@1 | Recall@4 | MRR@8 |
| --- | ---: | ---: | ---: |
| BM25 | 0.7318 | 0.9219 | 0.8296 |
| MiniLM cosine | 0.8255 | 0.9688 | 0.8951 |
| MiniLM cross-encoder | 0.8724 | 0.9688 | 0.9294 |

Both neural methods had Recall@8 of 1.0 and no reranker fallbacks. BM25's
development run is `reports/scifact-v2-development-bm25.json`. These are development results
in controlled candidate pools, not fixed-test results or verdict accuracy.

## Complete development verdict run

`reports/colab-development-verdicts-100.json` records the five-verdict Qwen
open-weight profile on all 100 development claims. Exact-label accuracy was
0.5000 (95% Wilson interval 0.4038-0.5962), macro F1 0.5842, with no missing
verdicts, infrastructure errors, or reranker fallbacks. There were no observed
false-supported predictions. Support recall was low: 17 of 47 supported claims
were labeled supported. Twenty-one predictions were weak support and 14 were
human review; both count as errors against SciFact's three expert labels.

The generic classifier below is a diagnostic comparison with a different label
policy. Neither result establishes autonomous publication safety.

## Fixed test retrieval

`reports/colab-test-retrieval-300.json` covers all 300 fixed test claims. Retrieval
metrics apply to the 188 claims with annotated evidence; 112 have no retrieval
target. Both neural strategies used identical cached embeddings and had no
reranker fallbacks.

| Method | Recall@1 | Recall@4 | MRR@8 |
| --- | ---: | ---: | ---: |
| BM25 | 0.6387 | 0.8183 | 0.7382 |
| MiniLM cosine | 0.6933 | 0.9225 | 0.8151 |
| MiniLM cross-encoder | 0.8001 | 0.9447 | 0.8881 |

Neural Recall@8 was 0.9854. These are controlled 16-abstract-pool results.

## Complete fixed-test verdict run

[`reports/colab-test-verdicts-300.json`](reports/colab-test-verdicts-300.json)
records all 300 fixed test claims using the frozen five-verdict Qwen profile.
It completed on October 27, 2026 on a free Colab Tesla T4.

| Metric | Result |
| --- | ---: |
| Exact-label accuracy, including missing outputs | 152/300 (50.67%) |
| Approximate 95% Wilson accuracy interval | 45.04%-56.28% |
| Macro F1 over the three observed gold labels | 0.6038 |
| Supported precision / recall | 88.24% / 36.29% |
| False-supported predictions against gold labels | 6 |
| Weak-support / human-review predictions | 68 / 38 |
| Missing verdicts / reranker fallbacks | 4 / 0 |

The four failed requests recorded `VerdictError` for C501, C814, C1241 and C1280.
They remain missing predictions in the 300-claim denominator. The 106 weak-support
and human-review outputs count as incorrect exact labels against SciFact's three
classes; they have not been remapped into correct answers. Five of the six
false-supported predictions have contradicted gold labels and one has an
unsupported gold label. See [error analysis and interpretation](ERROR_ANALYSIS.md).

The recovery reused only the verified first 50 cases from the executed notebook;
the old log-only cases 51-70 were not treated as saved results. The original
partial report and recovery hashes remain in
`reports/colab-test-verdicts-partial-50.json` and
`reports/colab-checkpoint-recovery.json`. The remaining 250 cases ran with the
same inference code, model revisions, package versions and test labels, with
Drive checkpoints every five cases. No test-time prompt or threshold tuning was
performed. The first 50 predictions are unchanged, and every final evidence
ranking matches the separate 300-case retrieval report.

`reports/colab-test-completion-verification.json` records the final report/archive
hashes and validation checks. The final archive SHA-256 is
`f0bf2f895d2e269fb9961b49919168b66499ed312e432bfef151a60271e4bc39`.

## Separate three-class NLI diagnostic

[`colab/nli_baseline.py`](../colab/nli_baseline.py) uses
[DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
on saved reranked evidence. It processes exact source spans using overlapping
token windows, with no silent input truncation. It is GPU-only and makes no
hosted inference API calls.

Eight development configurations compare top-one versus top-four evidence and
thresholds 0.5, 0.65, 0.8, and 0.9. Development macro F1 selected **top one,
threshold 0.5**. The script requires that frozen development configuration,
model revision, and code hash before test. Softmax values are not calibrated
truth probabilities. The model has only three classes and does not implement
the application's five-verdict policy or its safety checks.

| Split | Claims | Accuracy | Macro F1 | False-supported predictions |
| --- | ---: | ---: | ---: | ---: |
| Development | 100 | 0.7000 | 0.6806 | 8 |
| Fixed test | 300 | 0.6567 | 0.6474 | 22 |

The test accuracy interval is 0.6013-0.7081. Supported precision is 0.7684 and
supported recall 0.5887. Reports are `reports/colab-nli-development-100.json` and
`reports/colab-nli-test-300.json`. The code, data and retrieval-report hashes were
verified after export. `reports/colab-frozen-test-protocol.json` records both
profiles before test; `reports/colab-environment.json` records the runtime stack.

## Policy examples and application demo

The first policy-v2 run produced ten missing predictions: the free model returned
`C1` for an input claim named `PC1`, and the unknown-ID guard rejected it. Its
failed report remains in `reports/colab-policy-10-failed.json`. Policy-v3 keeps
the same ten source/claim/label examples and changes only the claim IDs to the
`C<number>` format issued by production extraction. That repeat scored 10/10,
with all five labels represented, in `reports/colab-policy-10.json`. These are
synthetic implementation examples, not independent quality evidence; do not
count the two versions as twenty distinct claims.

The live application demo extracted three claims, supported two sourced facts,
marked an invented advertising-revenue increase unsupported, and blocked
publication. Its inspectable result is `reports/colab-application-demo.json`.
This is a small integration check, not a replacement for domain evaluation.

## Reproduce the baseline

From `backend/`, using its virtual environment:

```bash
python benchmark.py run \
  --dataset ../evaluation/datasets/scifact-v2/test.json \
  --mode lexical \
  --output ../evaluation/reports/my-bm25-run.json
```

This requires no key or network. Choose a new output path for each run; existing
reports are protected from overwrite. `--limit 10` produces a report explicitly
marked as partial. The development run uses `development.json` instead.

## Compare the actual model pipeline

Configure `backend/.env` with working Qwen access and use development first:

```bash
python benchmark.py run \
  --dataset ../evaluation/datasets/scifact-v2/development.json \
  --mode compare --verdicts \
  --output ../evaluation/reports/my-development-model-run.json
```

Then freeze code/prompts and run test without changing them based on test scores:

```bash
python benchmark.py run \
  --dataset ../evaluation/datasets/scifact-v2/test.json \
  --mode compare --verdicts \
  --output ../evaluation/reports/my-test-model-run.json
```

`cosine` is the embedding-only baseline; `rerank` uses the production neural
reranker plus its lexical rescue rule. `compare` uses identical cached embeddings
and cases for both. This measures the combined reranking policy, not a pure
neural-only ablation. `--workers` controls concurrency (default four). Omit
`--verdicts` to measure retrieval only. Public-data embeddings are stored under
ignored `evaluation/cache/`; application uploads never enter this cache.

For policy examples, use `--dataset ../evaluation/datasets/policy-v3.json
--mode lexical --verdicts` with a new output filename. This isolates verdict
behavior without embedding calls.

Reports record dataset hash, code hashes, Git base revision (local uncommitted
changes may exist), UTC timestamp, model configuration, per-claim rankings,
quotes, citations, confusion matrix, support counts, per-label precision/recall/F1,
observed-label macro F1, missing verdicts, and false-supported predictions.
Missing outputs are incorrect; provider errors remain visible. Reranker fallback
claims are counted separately, so a degraded run cannot masquerade as a clean
reranker evaluation. First-stage selection and ranking use no gold labels.

Accuracy includes an approximate 95% Wilson interval over claims, which is
not a guarantee of independent samples or production performance. Verdict
classes absent from a dataset have null recall/F1. Retrieval timing excludes
embedding preparation; verdict timing is for batches of up to six claims.
Neither timing is application end-to-end latency. Use `wall_seconds` for total
benchmark runtime including embedding preparation.

## Rebuild the pinned dataset

Download the official archive to `evaluation/external/scifact-data.tar.gz` from:
<https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz>.

From `backend/`:

```bash
python benchmark.py build-scifact \
  --archive ../evaluation/external/scifact-data.tar.gz \
  --output ../evaluation/datasets/scifact-v2
```

The importer checks the pinned archive SHA-256 before reading known members.
The download, embeddings, and provider calls are separate from normal CI. CI
validates the checked-in datasets and exercises application behavior using mocks.

## Limits

- This public benchmark may be in a model's pretraining data. It is held apart
  from our development work, not guaranteed unseen by the model.
- Research abstracts differ from civic newsroom documents. A future independently
  annotated newsroom evaluation is needed before claiming that domain's accuracy.
- Gold evidence is present in each evidence-bearing pool. This does not establish
  full-corpus retrieval quality or missing-document detection.
- No annotated evidence means unsupported under this protocol, not false in the
  real world. Exact quote matching validates provenance, not semantic entailment.
- Claim extraction, PDF parsing, privacy checks, and publication controls have
  integration/regression tests; they are not measured by SciFact verdict scores.

## Historical smoke test

The June 29, 2026 aggregate `results.json` reports 12/12 verdict accuracy,
12/12 Recall@4, and 6/12 Recall@1 with `qwen3.6-flash`. It has no per-claim
artifacts and has not been regenerated. Each old case contains six passages,
so first-stage candidate pruning was not tested. `clinical_trial/C3` also has
a disputable gold label: a nonsignificant result does not prove equivalence or
inferiority. Keep these numbers historical; do not use them as the headline
project result. `python evaluate.py` remains available for that old format.
