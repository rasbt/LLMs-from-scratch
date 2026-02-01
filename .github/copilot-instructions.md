# Copilot instructions for this repo

## Big picture structure
- This repo mirrors the book chapters; most “source of truth” lives in chapter notebooks under chXX/01_main-chapter-code and related bonus folders. Examples: [ch04/01_main-chapter-code/ch04.ipynb](ch04/01_main-chapter-code/ch04.ipynb) and [ch05/01_main-chapter-code/gpt_train.py](ch05/01_main-chapter-code/gpt_train.py).
- Reusable, importable code is exposed via the optional PyPI package under pkg/llms_from_scratch; module names map to chapters (see [pkg/llms_from_scratch/README.md](pkg/llms_from_scratch/README.md)).
- Bonus/variant implementations live in chapter subfolders (e.g., KV cache in [ch04/03_kv-cache](ch04/03_kv-cache), Llama/Qwen/Gemma variants in [ch05](ch05)). Keep these scoped; do not mix with main chapter code unless explicitly requested.
- This repo is tied to a print book; main chapter code should remain aligned with the book’s presentation. Prefer minimal, surgical edits and avoid broad refactors in chapter notebooks/scripts.

## Workflows and commands (from CI)
- Python deps are primarily in requirements.txt and are installed with pip/uv. Quickstart: pip install -r requirements.txt (see [setup/README.md](setup/README.md)).
- CI uses uv or pixi to run targeted pytest suites and nbval notebook checks. Representative commands (see [/.github/workflows/basic-tests-linux-uv.yml](.github/workflows/basic-tests-linux-uv.yml)):
  - uv sync --dev
  - pytest ch04/01_main-chapter-code/tests.py
  - pytest --nbval ch02/01_main-chapter-code/dataloader.ipynb
- Style checks run ruff (flake8 replacement) across the repo (see [/.github/workflows/pep8-linter.yml](.github/workflows/pep8-linter.yml)).

## Repo-specific patterns
- Notebooks often import sibling Python files in the same chapter folder; keep local relative paths stable when editing notebooks or scripts.
- Tests are scattered per chapter and bonus folder (e.g., [ch05/07_gpt_to_llama/tests](ch05/07_gpt_to_llama/tests)); update or add tests next to the code you touch.
- Optional features (UIs, dataset utilities, alternative architectures) live in dedicated subfolders; prefer adding new experiments in the relevant bonus area rather than modifying main chapter notebooks.

## Integration points
- Packaging: pkg/llms_from_scratch exposes chapter APIs for external use; changes there affect users installing the PyPI package.
- External deps include PyTorch and optional libraries (transformers, nbval, etc.); CI installs extras for specific test folders (see [/.github/workflows/basic-tests-linux-uv.yml](.github/workflows/basic-tests-linux-uv.yml)).
