# Contributing

Issues and small, focused pull requests are welcome. Please include a minimal fixture or test for changes to a deterministic checker: those checks are deliberately conservative and a locally reasonable rule can damage a different caption pattern. Do not commit model weights, generated datasets, private source images, tokens, cluster paths, or paper PDFs.

Run `pytest -q` before opening a pull request. For prompt changes, include before/after examples and report both parse rate and correspondence errors rather than only showing the best generations.
