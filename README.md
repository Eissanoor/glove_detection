## Glove Detection System

A professional, end-to-end glove vs no_glove detection system using YOLOv8, with:
- Training on your dataset in `dataset/glove/`
- Inference CLI for images/videos/webcam
- Streamlit UI for quick testing

### Setup
1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

2. Install PyTorch first (to avoid DLL issues):
```bash
# For CPU-only (recommended to avoid DLL issues):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# OR if you have CUDA GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. Install other dependencies:
```bash
pip install -r requirements.txt
```

### Train (TensorFlow - No PyTorch DLL issues)
```bash
python -m src.train_tf --data dataset/glove/data.yaml --epochs 50 --img 640
```

### Inference CLI (TensorFlow)
```bash
python -m src.infer_tf --source path/to/image_or_video --weights runs/detect/glove_model.h5
```

### Alternative: PyTorch (if you fix DLL issues)
```bash
# First install PyTorch properly:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then train:
python -m src.train --data dataset/glove/data.yaml --epochs 50 --img 640

# Inference:
python -m src.infer --source path/to/image_or_video --weights runs/detect/train/weights/best.pt
```

### Streamlit UI
```bash
streamlit run src/ui_app.py
```

### Project Structure
- `src/config.py`: paths and defaults
- `src/train.py`: training entrypoint
- `src/infer.py`: inference CLI
- `src/ui_app.py`: Streamlit user interface
