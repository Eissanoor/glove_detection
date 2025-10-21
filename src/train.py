import argparse
from pathlib import Path

from ultralytics import YOLO

from config import DATA_YAML, DEFAULT_BATCH, DEFAULT_EPOCHS, DEFAULT_IMAGE_SIZE, DEFAULT_MODEL, ensure_dirs


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train YOLOv8 for glove detection")
	parser.add_argument("--data", type=str, default=str(DATA_YAML), help="path to data.yaml")
	parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="base model weights")
	parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="number of epochs")
	parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="batch size")
	parser.add_argument("--img", type=int, default=DEFAULT_IMAGE_SIZE, help="image size")
	parser.add_argument("--device", type=str, default=None, help="cuda device like '0', '0,1', or cpu")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	ensure_dirs()

	model = YOLO(args.model)
	results = model.train(
		data=args.data,
		epochs=args.epochs,
		batch=args.batch,
		imgsz=args.img,
		device=args.device,
	)
	print(results)


if __name__ == "__main__":
	main()
