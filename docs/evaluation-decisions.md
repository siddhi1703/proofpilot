# Evaluation decisions

Recorded October 26, 2026, before the project test model run.

The original hosted Qwen account has no usable free quota. The user requested
free cloud computation without running models on their laptop. Evaluation uses
one Google Colab Tesla T4 and public open weights, with no paid API fallback.

## Model profile

| Component | Model | Resolved revision |
| --- | --- | --- |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Reranker | cross-encoder/ms-marco-MiniLM-L6-v2 | `233902d25c440f23af6f7d6e94d2946bac0bee0a` |
| Verifier | Qwen/Qwen3-4B-Instruct-2507 | `cdbee75f17c01a7cc42f958dc650907174af0554` |

The verifier uses NF4 four-bit weights, float16 computation, double quantization,
greedy generation, a 12,000-token input budget, and a 2,048-token output cap.
Long abstracts use overlapping windows. This profile replaces the hosted model
transports while preserving production verdict validation and scoring.

## Comparison protocol

1. Run a ten-case development pilot comparing cosine and reranked evidence,
   including both verifier passes. Preserve this partial result.
2. Compare both retrieval strategies on all 100 development claims. Run the
   verifier once per claim on reranked evidence for the larger evaluation.
3. Inspect development outcomes, then freeze the same inference implementation
   and model profile before the separate 300-case project test.
4. Compare retrieval methods on all test cases, and score the verifier once on
   reranked evidence. This does not measure a verifier accuracy gain from reranking.
5. Keep the ten synthetic policy cases separate from the expert-labeled dataset.

The pilot and larger run use different runner fingerprints because the runner
gained an explicit single-strategy option. The retrieval algorithm, verdict
prompt, model profile and validation logic were not tuned after the pilot.
The current larger development runner ID is `0e967755e229a610ab53`; the pilot ID
is `e03bfea2b211846fc807`. Reports record the full configuration and code hash.

The full development result was 50% exact-label accuracy, with low support
recall and no observed false-supported predictions. A separate three-class
DeBERTa NLI diagnostic was therefore added before test. Its model revision is
`6f5cf0a2b59cabb106aca4c287eed12e357e90eb`; its source hash is
`7c2f677d102b3a459e5539195f0c63c81ef1c2fcae14929e7c90a429391346ed`.
Eight predetermined development settings selected top-one evidence and a 0.5
threshold by macro F1. The baseline does not implement five verdicts or the
application safety checks; it is a diagnostic, not an automatic-approval policy.

Both profiles were frozen at **2026-10-26 22:38:08 UTC**, before test inference,
in `evaluation/reports/colab-frozen-test-protocol.json`. The Qwen inference source
hash is `b03d98393f63fa83d651d48addfd2fc9ee7bd17534d3f15ca6a54c70736a64fe`.
The NLI test completed at 65.67% accuracy with 22 false-supported predictions.
No inference or threshold changes were made after observing test outcomes.

Missing verdicts count as incorrect. Validation-driven human review and weak
support predictions also count as incorrect against SciFact's three labels.
Do not remap them to improve accuracy, remove difficult cases, or silently retry
failed chunks. A conservative system may have fewer false approvals and lower
recall; publish both the confusion matrix and failure counts.

## Scope

Gold abstracts are included in shuffled 16-abstract pools. Retrieval is therefore
controlled candidate-pool retrieval. Unsupported means no annotated evidence
under the upstream protocol; distractor annotations are not exhaustive.
Public SciFact data may have appeared in model pretraining. The test split is
upstream development, separated from this project's development work; it is not
the unreleased official SciFact test. These results do not establish newsroom
accuracy, claim-extraction accuracy, or safe autonomous publication.

## Storage

Google Drive mounting failed despite approval. Native Colab session upload is
used instead. Each five-case chunk is saved in runtime storage, and checkpoint
archives and reports are copied to the user's chosen Drive folder. Temporary
runtime storage alone is not a durable backup.
