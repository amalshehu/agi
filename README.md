# ARC Prize Tools

This repository collects a set of experiments for working with the ARC Prize dataset. The focus is on
rule based solvers with a few lightweight neural components.  It is **not** a general artificial intelligence system.

## Repository layout

- `core/` – contains the `CognitiveAgent` used by the solvers
- `arc_prize_pipeline.py` – helper for running solvers on ARC tasks
- `data_loader.py` – loader for the JSON datasets under `arc-prize-2025`
- `test_*` – small scripts demonstrating solver behaviour

## Getting started

1. Ensure the `arc-prize-2025` data directory is present in the repository root.
2. Use `python arc_prize_pipeline.py` to run the solver pipeline.
3. See `simple_score.py` for evaluating a JSON submission file.

The code is a research prototype and many components are simplified.

