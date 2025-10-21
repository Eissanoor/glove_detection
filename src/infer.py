import argparse
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO

from config import CONF_THRESHOLD_DEFAULT, IOU_THRESHOLD_DEFAULT


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run YOLOv8 inference for glove detection")
	parser.add_argument("--source", type=str, required=True, help="image/video file, directory, or webcam index (0)")
	parser.add_argument("--weights", type=str, required=True, help="path to trained weights, e.g., best.pt")
	parser.add_argument("--conf", type=float, default=CONF_THRESHOLD_DEFAULT, help="confidence threshold")
	parser.add_argument("--iou", type=float, default=IOU_THRESHOLD_DEFAULT, help="NMS IoU threshold")
	parser.add_argument("--show", action="store_true", help="display results in a window")
	parser.add_argument("--save", action="store_true", help="save visualized results next to inputs")
	return parser.parse_args()


def run_on_image(model: YOLO, image_path: Path, conf: float, iou: float, show: bool, save: bool) -> None:
	result = model.predict(source=str(image_path), conf=conf, iou=iou, verbose=False)[0]
	plot = result.plot()
	if save:
		out_path = image_path.with_name(image_path.stem + "_pred" + image_path.suffix)
		cv2.imwrite(str(out_path), plot)
	if show:
		cv2.imshow("Prediction", plot)
		cv2.waitKey(0)


def run_on_video(model: YOLO, source: str, conf: float, iou: float, show: bool, save: bool) -> None:
	cap = cv2.VideoCapture(0 if source.isdigit() else source)
	writer: Optional[cv2.VideoWriter] = None
	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			result = model.predict(source=frame, conf=conf, iou=iou, verbose=False)[0]
			plot = result.plot()
			if show:
				cv2.imshow("Prediction", plot)
				if cv2.waitKey(1) & 0xFF == 27:
					break
			if save and not source.isdigit():
				if writer is None:
					fourcc = cv2.VideoWriter_fourcc(*"mp4v")
					out_path = Path(source).with_name(Path(source).stem + "_pred.mp4")
					writer = cv2.VideoWriter(str(out_path), fourcc, cap.get(cv2.CAP_PROP_FPS) or 30.0, (plot.shape[1], plot.shape[0]))
				writer.write(plot)
	finally:
		cap.release()
		if writer is not None:
			writer.release()
		cv2.destroyAllWindows()


def main() -> None:
	args = parse_args()
	model = YOLO(args.weights)

	source_path = Path(args.source)
	if args.source.isdigit() or source_path.suffix.lower() in SUPPORTED_VIDEO_EXTS:
		run_on_video(model, args.source, args.conf, args.iou, args.show, args.save)
		return

	if source_path.is_dir():
		for image_file in sorted(source_path.iterdir()):
			if image_file.suffix.lower() in SUPPORTED_IMAGE_EXTS:
				run_on_image(model, image_file, args.conf, args.iou, args.show, args.save)
		return

	if source_path.is_file() and source_path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
		run_on_image(model, source_path, args.conf, args.iou, args.show, args.save)
		return

	raise ValueError("Unsupported --source. Provide image/video file, directory, or webcam index like '0'.")


if __name__ == "__main__":
	main()
