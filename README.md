# ARC Prize Tools

This repository collects a set of experiments for working with the ARC Prize dataset. The focus is on
rule based solvers with a few lightweight neural components.  It is **not** a general artificial intelligence system.

## Repository layout

- `core/` – simple neural and symbolic building blocks
- `arc_prize_pipeline.py` – helpers for running solvers on ARC tasks
- `training_pipeline.py` – example routine to train the small neural models
- `data_loader.py` – loader for the JSON datasets under `arc-prize-2025`
- `tests/` and `test_*` – short scripts demonstrating solver behaviour

## Getting started

1. Ensure the `arc-prize-2025` data directory is present in the repository root.
2. (Optional) run `python training_pipeline.py` to train the neural modules.
3. Use `python arc_prize_pipeline.py` to run the solver pipeline.
4. See `simple_score.py` for evaluating a JSON submission file.

The code is a research prototype and many components are simplified.

