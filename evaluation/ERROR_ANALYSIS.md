# Fixed-test error analysis

The frozen Qwen3-4B five-verdict profile completed all 300 SciFact project-test
claims. It matched 152 gold labels (50.67%, macro F1 0.6038), with four failed
verdict requests retained as missing predictions. Source data, model revisions,
code, labels and all per-claim outputs are preserved in the
[full report](reports/colab-test-verdicts-300.json). No test errors were removed
or relabeled, and no prompts were changed after inspecting the test.

## Retrieval is not entailment

Reranking recovered 94.47% of labeled abstracts in the top four, averaged over
188 evidence-bearing claims. Yet all five false-supported predictions with
contradicted gold had their annotated abstract at rank one. Retrieval therefore
did not prevent these verdict errors. Exact quote validation verifies that words occur in a source;
it cannot prove that the source supports the claim's direction or scope.

| Case | Recorded failure |
| --- | --- |
| C249 | The claim says energy expenditure decreases, while the model's own quote says it increases. The output was supported despite contradicted gold. |
| C1339 | The claim says traumatic procedures increase, while the quoted result says their risk decreases. The output was supported despite contradicted gold. |
| C759 | The model cited a different candidate abstract's statement about potential transmission reduction, despite contradicted gold and the annotated abstract ranking first. |

These are benchmark interpretations of supplied scientific text, not medical
recommendations. Five false-supported predictions have contradicted gold and
one has unsupported gold. The latter, C1099, cites a mechanism statement about
cholesterol biosynthesis rather than the claimed blood-cholesterol outcome.
This illustrates possible mechanism-versus-outcome and annotation-coverage
ambiguity: unsupported denotes upstream insufficient-evidence annotation, not
a declaration that the claim is medically false. Distractor abstracts were not
exhaustively reannotated. All original gold labels remain unchanged.

## Conservative outputs also reduce coverage

Of 124 supported gold claims, only 45 were returned as supported: 36.29% recall.
Supported precision was 45/51 (88.24%). Across all claims, 68 outputs were weak
support and 38 required human review. SciFact has neither class, so these 106
outputs count as exact-label errors. They must not be silently mapped into
supported or unsupported to improve the score.

C501, C814, C1241 and C1280 recorded `VerdictError`. The report preserves the
error class and missing prediction, without claiming a more specific cause
than the captured diagnostics establish. Cached failures were not silently retried.

The separate three-class NLI diagnostic scored 65.67% accuracy but had 22
false-supported predictions. Its label policy and checks differ from the
application; this comparison does not justify replacing the review gate.

## Practical limits

The result supports a reproducible retrieval-and-verdict prototype with visible
failures. It does not establish autonomous publication safety. Final editor
review remains necessary. A future improvement should be developed against a
separate development set, then evaluated on a new untouched test set. Newsroom
documents, claim extraction, PDF parsing and full-corpus retrieval need their
own quality evaluations before broader deployment claims.
