# SciFact attribution and adaptation

Source: [Allen Institute for AI, SciFact](https://github.com/allenai/scifact).
Paper: David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen,
Arman Cohan, and Hannaneh Hajishirzi. *Fact or Fiction: Verifying Scientific
Claims*. EMNLP 2020. <https://arxiv.org/abs/2004.14974>.

Upstream [data documentation](https://github.com/allenai/scifact/blob/master/doc/data.md)
and [licenses](https://github.com/allenai/scifact/blob/master/LICENSE.md):

- Claim text and evidence annotations: CC BY 4.0,
  <https://creativecommons.org/licenses/by/4.0/>.
- Corpus abstracts from Semantic Scholar S2ORC: ODC-By 1.0,
  <https://opendatacommons.org/licenses/by/1-0/>.

These data are separately licensed and are not covered by ProofPilot's MIT
software license. Attribution to the original authors and S2ORC is retained here.

Adaptations: a deterministic 100-claim subset of upstream training data forms
our development set. All 300 upstream development claims form our project test
set. We map SUPPORT to supported, CONTRADICT to contradicted, and absent
annotated evidence to unsupported. We join abstract sentences into passages,
retain gold sentence groups, and add lexical hard negatives to 16-abstract pools.
The data are not the unreleased official SciFact test set and the scores are not
comparable to full-corpus leaderboard scores. No endorsement by Ai2 is implied.

Archive URL and checksum are pinned in `manifest.json` and the importer. The
rebuild reads known tar members without extracting paths. All candidate source
documents, including distractors, are disjoint between our two splits.
