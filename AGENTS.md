# Repository Guidelines

## Project Structure & Module Organization
Core scripts live at the repository root. `generate_dataset.py`, `load_dataset.py`, and `shapes.py` handle dataset synthesis; `model.py` defines the CNN; `train_model.py` plus `run_experiment.py` orchestrate training runs and checkpoint export. GUI logic is split across `explore_main.py`, `callbacks.py`, `ui_layout.py`, and `app_state.py`, while `view_dataset.py` provides a standalone browser. Generated artifacts follow the `DD-MM-YYYY_HH-MM_<N>imgs_<SIZE>x<SIZE>` pattern and should remain untracked.

## Environment Setup & Key Commands
Use Python 3.12 in a virtual environment: `python -m venv .venv` then `.\\.venv\\Scripts\\Activate.ps1`. Install runtime deps with `pip install pillow numpy tqdm matplotlib torch torchvision`. Create a dataset via `python generate_dataset.py`; for a pure palette invoke the inline example in `README.md`. Train with `python train_model.py`, or run the end-to-end workflow with `python run_experiment.py` (saves checkpoints and accuracy plots inside the dataset folder). Launch the exploration GUI with `python explore_main.py` and the lightweight viewer with `python view_dataset.py`.

## Coding Style & Naming Conventions
Follow PEP 8: 4-space indentation, snake_case for functions, CAPS for constants, and PascalCase only for Tkinter classes. Keep modules focused; utility helpers belong in `utils.py`, GUI state in `app_state.py`. Prefer explicit imports and type hints where practical. Document non-obvious logic with concise inline comments.

## Testing Guidelines
Automated tests are not yet present. Before opening a pull request, generate a small dataset, run `python train_model.py`, and confirm the reported accuracy curve is reasonable. For GUI changes capture screenshots or short clips demonstrating the new behavior. New utilities should include docstrings or inline assertions to aid future pytest adoption.

## Commit & Pull Request Guidelines
The current history lacks strong conventions; move toward imperative, descriptive commits (`Add dropout option to trainer`). Reference related datasets or artifacts when relevant and avoid committing `.zip`, `.pth`, or image outputs. Pull requests must describe goals, list runnable commands, attach GUI evidence when appropriate, and link to any tracking issue.

## Data & Artifact Handling
Large folders under `20-09-2025_*` exemplify expected outputs; keep them local or prune as needed. Update `.gitignore` if new artifact types appear, and document storage paths when sharing models or datasets with collaborators.
