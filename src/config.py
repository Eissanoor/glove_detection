from pathlib import Path

# Project root is the workspace directory containing this src folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset
DATASET_DIR = PROJECT_ROOT / "dataset" / "glove"
DATA_YAML = DATASET_DIR / "data.yaml"

# Training defaults
DEFAULT_IMAGE_SIZE = 640
DEFAULT_EPOCHS = 50
DEFAULT_BATCH = 16
DEFAULT_MODEL = "yolov8n.pt"  # start small; can switch to yolov8s.pt

# Outputs
RUNS_DIR = PROJECT_ROOT / "runs"
DETECT_DIR = RUNS_DIR / "detect"

# UI
CONF_THRESHOLD_DEFAULT = 0.25
IOU_THRESHOLD_DEFAULT = 0.45


def ensure_dirs() -> None:
	DETECT_DIR.mkdir(parents=True, exist_ok=True)
