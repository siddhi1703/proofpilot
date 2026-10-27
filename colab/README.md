# Free Colab evaluation

First run `python3 scripts/build_colab.py` from the repository root. This creates
a matched notebook and ZIP using the standard library; it does not install or
run models locally. Use both generated files together.

Use `ProofPilot_Free_Colab.ipynb` with `proofpilot-colab-bundle.zip` in the same
Google Drive folder, **My Drive / ProofPilot Colab**. The archive contains the
current local backend, public evaluation data, tests, and this runtime profile.
It deliberately excludes `.env`, API keys, model weights, Git history, private
uploads, dependencies, and existing frontend work.

1. Open the notebook in Google Colab using the Google account containing that folder.
2. Select **Runtime > Change runtime type > T4 GPU** and stay on the free tier.
3. Run the cells in order. For Drive storage, approve the Drive mount when Google
   asks. If it fails with a credential-propagation error, choose `USE_DRIVE = False`
   and use Colab's **Files > Upload to session storage** for the bundle.
4. The default model run evaluates only ten development cases, clearly marked
   as partial. Drive-mode reports and embeddings are checkpointed under
   `ProofPilot Colab/outputs`. Session mode uses `/content/proofpilot-session/outputs`.
   Export its checkpoint ZIP and save it in Drive before the runtime disconnects;
   temporary session files do not survive a recycled runtime.
5. After reviewing development failures, enable the full-development cell.
   Freeze the implementation before enabling the separate 300-case test cell.
   Each full run compares both retrieval methods and evaluates verdicts using
   reranked evidence once per claim. The pilot compares verdicts for both methods.

This runs open weights on the Colab GPU: MiniLM embeddings, a MiniLM cross-encoder,
and Qwen3-4B-Instruct-2507 with NF4 four-bit inference. No paid model inference
endpoint is contacted, and no API key is needed. Public model files are downloaded
inside the Colab runtime; the first download can be several GB. The notebook does
not upgrade your Colab plan, buy compute, create cloud services, or run local models
on your Mac.

MiniLM normally truncates long text. This profile instead encodes all overlapping
token windows and averages their normalized vectors. Reranking takes the maximum
cross-encoder score across source windows. Those choices and exact model revisions
are recorded. They differ from the hosted Qwen architecture; do not describe their
scores as results from the original cloud models. The same production verdict
validation and benchmark scoring remain in use.

The runner saves a checkpoint after every five cases, using a run identity derived
from the code, dataset, and resolved model revisions. A repeated run reuses completed
checkpoints. If disconnected within a chunk, at most that chunk needs repeating.
For session-mode recovery, upload the exported checkpoint ZIP into a new Colab
runtime and use the notebook's **Restore a saved run** cell before restarting
the experiment. It checks paths, JSON and available manifest hashes, and refuses
to overwrite a different existing result. Older archives remain supported;
rebuildable vector caches are skipped. New backup archives contain JSON results
and a checksum manifest, which keeps downloads small.
Do not select the best of repeated test runs. Failed verdicts stay in the score and
cached errors are not silently retried. Changed packages are rejected against an
existing run; use a separately named experiment folder for a changed software stack.

Free GPU allocation and session length are not guaranteed. If Google cannot
allocate a GPU, stop and retry later; there is no paid API fallback. This notebook
is for interactive evaluation and a small in-notebook demo, not public web hosting.
Model scores must come from recorded GPU runs. The first ten-case development
pilot and full development run have been run on a free Tesla T4. The complete
development verdict accuracy was 50%, with low support recall and no observed
false-supported predictions. The fixed test retrieval comparison and a separate
three-class NLI test have also completed. The final five-verdict Qwen test report
has not been recovered; the last confirmed checkpoint was 40/300. See
`evaluation/README.md` for the actual scores and limitations.

Model sources and licenses:

- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache 2.0.
- [MS MARCO MiniLM reranker](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), Apache 2.0.
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), Apache 2.0.
- The separate [DeBERTa NLI diagnostic](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
  uses MIT-licensed weights. It has three labels and does not implement application safety policy.
- SciFact attribution and separate data licenses are in `evaluation/datasets/scifact-v2/ATTRIBUTION.md`.

The notebook and checkpoint logic are checked locally without downloading model
weights. GPU inference must be verified in Colab; local tests do not establish its
model accuracy or runtime compatibility with every future Colab image.

## Reproduce the separate NLI diagnostic

After setup and the complete retrieval comparisons, run this in the Colab
runtime. Use new output names for a new experiment; existing results are
protected from overwrite. The test requires the saved development selection.

```python
import importlib.util
spec = importlib.util.spec_from_file_location('proofpilot_nli', PROJECT / 'colab/nli_baseline.py')
nli_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nli_module)
nli = nli_module.NLIDiagnostic()  # GPU only; weights download inside Colab
development_result = nli.run(
    PROJECT / 'evaluation/datasets/scifact-v2/development.json',
    OUTPUTS / 'retrieval-development-100.json', OUTPUTS / 'nli-development-100.json')
test_result = nli.run(
    PROJECT / 'evaluation/datasets/scifact-v2/test.json',
    OUTPUTS / 'retrieval-test-300.json', OUTPUTS / 'nli-test-300.json',
    development_report=OUTPUTS / 'nli-development-100.json')
```

This three-class diagnostic uses saved rankings, overlapping exact-source
windows, and a frozen development threshold. It is separate from the application's
five-verdict policy and safety checks. Its higher exact-label development accuracy
also came with false-supported predictions; do not use it to approve publication.

## Recover an executed notebook

If a Colab session is gone but the downloaded notebook contains a backup
textarea, recover it locally without executing any notebook code:

```bash
python3 scripts/build_resume.py \
  --notebook /path/to/ProofPilot_Free_Colab.ipynb \
  --output colab/results/my-recovery
```

The standard-library builder verifies the embedded ZIP's size, SHA-256 and CRC,
restores it safely, checks the frozen source and dataset hashes, and compares
completed chunks with their combined report and pinned gold labels. It creates
a recovery notebook plus matching source/checkpoint ZIPs. The notebook pins
the original model revisions and software versions, reuses completed inference,
and saves subsequent progress to Drive by default. An incompatible Torch runtime
is rejected rather than mixed with the saved experiment.

The October 27 recovery restored 50 portable test cases; the notebook log showed
70 cases, but the additional twenty had no portable backup. Only the verified
50 are reused. Their 48% exact-label accuracy is a partial result, not the final
300-case score. See `evaluation/reports/colab-checkpoint-recovery.json` and
`evaluation/reports/colab-test-verdicts-partial-50.json`.
