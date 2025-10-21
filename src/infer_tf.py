"""
TensorFlow-based inference for glove detection
Avoids PyTorch DLL issues
"""

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import CONF_THRESHOLD_DEFAULT


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class GloveDetector:
    def __init__(self, model_path: str, img_size: int = 640):
        self.model = keras.models.load_model(model_path)
        self.img_size = img_size
        self.classes = ['no_glove', 'glove']
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Resize to model input size
        img = cv2.resize(image, (self.img_size, self.img_size))
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize
        img = img.astype('float32') / 255.0
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(self, image, conf_threshold=0.5):
        """Predict glove/no_glove on image"""
        processed = self.preprocess_image(image)
        predictions = self.model.predict(processed, verbose=0)
        
        # Get class with highest probability
        class_id = np.argmax(predictions[0])
        confidence = predictions[0][class_id]
        
        if confidence > conf_threshold:
            return {
                'class': self.classes[class_id],
                'confidence': float(confidence),
                'class_id': class_id
            }
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Run TensorFlow inference for glove detection")
    parser.add_argument("--source", type=str, required=True, help="image/video file, directory, or webcam index (0)")
    parser.add_argument("--weights", type=str, required=True, help="path to trained model, e.g., glove_model.h5")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD_DEFAULT, help="confidence threshold")
    parser.add_argument("--show", action="store_true", help="display results in a window")
    parser.add_argument("--save", action="store_true", help="save visualized results next to inputs")
    return parser.parse_args()


def draw_bounding_box(image, class_name, confidence):
    """Draw bounding box and label on image"""
    result_image = image.copy()
    
    # Draw bounding box for no_glove detections (red square box)
    if class_name == "no_glove":
        height, width = image.shape[:2]
        # Calculate bounding box size (20% of image size)
        box_size = min(width, height) // 5
        # Center the box
        x1 = (width - box_size) // 2
        y1 = (height - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        # Draw red square bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # Draw corner markers for better visibility
        corner_size = 20
        # Top-left corner
        cv2.rectangle(result_image, (x1, y1), (x1 + corner_size, y1 + 3), (0, 0, 255), -1)
        cv2.rectangle(result_image, (x1, y1), (x1 + 3, y1 + corner_size), (0, 0, 255), -1)
        # Top-right corner
        cv2.rectangle(result_image, (x2 - corner_size, y1), (x2, y1 + 3), (0, 0, 255), -1)
        cv2.rectangle(result_image, (x2 - 3, y1), (x2, y1 + corner_size), (0, 0, 255), -1)
        # Bottom-left corner
        cv2.rectangle(result_image, (x1, y2 - 3), (x1 + corner_size, y2), (0, 0, 255), -1)
        cv2.rectangle(result_image, (x1, y2 - corner_size), (x1 + 3, y2), (0, 0, 255), -1)
        # Bottom-right corner
        cv2.rectangle(result_image, (x2 - corner_size, y2 - 3), (x2, y2), (0, 0, 255), -1)
        cv2.rectangle(result_image, (x2 - 3, y2 - corner_size), (x2, y2), (0, 0, 255), -1)
    
    # Draw label
    label = f"{class_name}: {confidence:.2f}"
    color = (0, 0, 255) if class_name == "no_glove" else (0, 255, 0)
    cv2.putText(result_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return result_image

def run_on_image(detector: GloveDetector, image_path: Path, conf: float, show: bool, save: bool):
    """Run detection on single image"""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    result = detector.predict(image, conf)
    
    # Draw result on image
    if result:
        image = draw_bounding_box(image, result['class'], result['confidence'])
        print(f"Detection: {result['class']}: {result['confidence']:.2f}")
    else:
        cv2.putText(image, "No glove detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("No glove detected")
    
    if save:
        out_path = image_path.with_name(image_path.stem + "_pred" + image_path.suffix)
        cv2.imwrite(str(out_path), image)
        print(f"Saved result to: {out_path}")
    
    if show:
        cv2.imshow("Glove Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_on_video(detector: GloveDetector, source: str, conf: float, show: bool, save: bool):
    """Run detection on video or webcam"""
    cap = cv2.VideoCapture(0 if source.isdigit() else source)
    writer: Optional[cv2.VideoWriter] = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = detector.predict(frame, conf)
            
            # Draw result on frame with bounding box tracking
            if result:
                frame = draw_bounding_box(frame, result['class'], result['confidence'])
            else:
                cv2.putText(frame, "No glove detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if show:
                cv2.imshow("Glove Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
            
            if save and not source.isdigit():
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out_path = Path(source).with_name(Path(source).stem + "_pred.mp4")
                    writer = cv2.VideoWriter(str(out_path), fourcc, cap.get(cv2.CAP_PROP_FPS) or 30.0, 
                                           (frame.shape[1], frame.shape[0]))
                writer.write(frame)
    
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    
    # Load model
    try:
        detector = GloveDetector(args.weights)
        print(f"Model loaded from {args.weights}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    source_path = Path(args.source)
    
    # Handle different source types
    if args.source.isdigit() or source_path.suffix.lower() in SUPPORTED_VIDEO_EXTS:
        run_on_video(detector, args.source, args.conf, args.show, args.save)
    elif source_path.is_dir():
        for image_file in sorted(source_path.iterdir()):
            if image_file.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                run_on_image(detector, image_file, args.conf, args.show, args.save)
    elif source_path.is_file() and source_path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
        run_on_image(detector, source_path, args.conf, args.show, args.save)
    else:
        raise ValueError("Unsupported --source. Provide image/video file, directory, or webcam index like '0'.")


if __name__ == "__main__":
    main()
