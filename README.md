# CNN Geometric Shapes

A toolkit for generating synthetic geometric-shape datasets, training a simple CNN classifier, and exploring the learned activations through a Tkinter GUI.

## Features
- **Dataset generator** (`generate_dataset.py`): create zipped image/label datasets with either random color palettes or a constrained pure RGB/black palette.
- **Training pipeline** (`train_model.py` & `run_experiment.py`): trains a small CNN (PyTorch) with configurable epochs, dropout, and batch size; saves checkpoints and accuracy plots.
- **GUI explorer** (`explore_main.py`): inspect dataset samples, predictions, activation maps, and kernel weights, now with per-image RGB readouts and kernel weight sums.
- **Dataset viewer** (`view_dataset.py`): lightweight Tkinter viewer for zipped datasets.

## Requirements
- Python 3.12 (or compatible)
- Recommended: virtual environment (`python -m venv .venv`)
- Packages installed via `pip install -r requirements.txt` (generate manually: see Installed packages section below)

Installed packages currently used:
```
pillow
numpy
tqdm
matplotlib
torch
torchvision
```

## Quick Start

### 1. Set up the environment
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install pillow numpy tqdm matplotlib torch torchvision
```

### 2. Generate a dataset
```powershell
python generate_dataset.py
```
The script creates a timestamped folder like `DD-MM-YYYY_HH-MM_<N>imgs_<SIZE>x<SIZE>` containing `images.zip`, `labels.zip`, and `shape_ids.txt`.

To use the pure-color palette:
```powershell
python - <<'PY'
from generate_dataset import generate_data
generate_data(num_images=40000, img_size=42, color_mode='pure')
PY
```

### 3. Train a model
```powershell
python train_model.py
```
Or run the combined workflow:
```powershell
python run_experiment.py
```
`run_experiment.py` now trains for 40 epochs by default; checkpoints (`.pth`) and accuracy plots (`*_accuracy.png`) are saved in the dataset folder.

### 4. Explore the model
Launch the GUI:
```powershell
python explore_main.py
```
- Select a dataset folder and corresponding `.pth` model.
- Browse images with prev/next buttons; the filter selection stays persistent.
- The left panel shows: true/predicted labels and dominant shape/background RGB values.
- Activation panel updates automatically; clicking a filter tile displays details and the kernel panel shows channel matrices with weight sums.

### 5. Extra viewer
```powershell
python view_dataset.py
```
Displays zipped datasets, supports zoom, grid overlay, and keyboard navigation.

## Project Structure
```
callbacks.py          # GUI logic & activation visualization helpers
ui_layout.py           # Tkinter layout for exploration app
app_state.py           # Shared state across callbacks
generate_dataset.py    # Dataset generation (random or pure palettes)
train_model.py         # Training loop and accuracy plotting
run_experiment.py      # Dataset generation + training pipeline
load_dataset.py        # PyTorch Dataset reading zipped archives
model.py               # Simple CNN architecture
utils.py               # Helper functions (datasets, visualizations, etc.)
view_dataset.py        # Standalone dataset viewer
explore_main.py        # GUI entry point
shapes.py              # Shape drawing primitives (rectangle, ellipse, etc.)
```

## Tips
- GUI requires a trained model and matching dataset folder. Ensure `shape_ids.txt` aligns with training dataset.
- Long training runs (e.g., 40 epochs on 40k images) can take ~25 minutes on CPU; consider GPU if available.
- Generated datasets and models are ignored by git (`.gitignore` excludes timestamped folders, `.zip`, `.pth`, accuracy plots, and `.venv`).
- You can safely delete folders like `20-09-2025_*` if you need to reclaim disk space.

## License
Add your chosen license here.
