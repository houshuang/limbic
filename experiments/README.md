# experiments/

The working files behind the numbers in [`../docs/EVIDENCE.md`](../docs/EVIDENCE.md).
Read the evidence page first — it has the findings. This directory is here so a
claim can be traced back to the script that produced it.

**These are historical and most of them no longer run as written.** Two
reasons, both worth knowing before you try:

1. **The package was renamed.** Every script here still says `from amygdala
   import …`, which was the name before the library became `limbic` with
   `amygdala` as a sub-package. The import that works today is `from
   limbic.amygdala import …`. The scripts were left at the name they were run
   under rather than rewritten to a state they were never executed in; the
   leftover `amygdala.egg-info/` you may see in a working copy is from the same
   era and is not tracked.
2. **Most read a corpus that is not in this repo** — private evaluation
   databases (`$AMYGDALA_EVAL_DB`, a local chat-search index, production claim
   sets). A few were partly genericised at some point, so a filename may name a
   project the file's body no longer does.

The public-benchmark scripts (`eval_stsb.py`, `eval_qqp_novelty.py`,
`eval_scifact_search.py`) fetch their own data and need only the import fixed.

| | |
|---|---|
| `expNN_*.py` | numbered experiments 1–21, matching the table in EVIDENCE.md |
| `exp_knowledge_map_matrix.py`, `knowledge_map_simulation.py` | row 23 |
| `document_similarity_design.md`, `calibration_document_similarity.md` | row 22 |
| `eval_*.py` | public-benchmark evaluations |
| `autoresearch_*` | parameter sweeps |
| `results/` | raw JSON output, including sample text from the evaluated corpora |
| `graphs/` | knowledge-map graphs used by `probe_ui.py` |

Nothing here is imported by the library, and nothing here is covered by the
test suite.
