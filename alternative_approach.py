"""
Alternative approach: Use OpenCV DNN with pre-trained models
This avoids PyTorch DLL issues entirely
"""

import cv2
import numpy as np
from pathlib import Path

class GloveDetector:
    def __init__(self, weights_path=None):
        """
        Initialize detector with OpenCV DNN
        This approach avoids PyTorch DLL issues
        """
        self.net = None
        self.classes = ['glove', 'no_glove']
        
        if weights_path and Path(weights_path).exists():
            # Load YOLO model using OpenCV DNN
            self.load_model(weights_path)
    
    def load_model(self, weights_path):
        """Load YOLO model using OpenCV DNN"""
        try:
            # This would work with ONNX exported models
            self.net = cv2.dnn.readNetFromONNX(weights_path)
            print(f"Model loaded from {weights_path}")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("You need to export your trained model to ONNX format first")
    
    def detect(self, image, conf_threshold=0.25):
        """Detect gloves in image"""
        if self.net is None:
            return [], image
        
        # Preprocess image
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Run inference
        outputs = self.net.forward()
        
        # Process detections
        detections = []
        h, w = image.shape[:2]
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > conf_threshold:
                    x_center, y_center, width, height = detection[:4]
                    x = int((x_center - width/2) * w)
                    y = int((y_center - height/2) * h)
                    w_box = int(width * w)
                    h_box = int(height * h)
                    
                    detections.append({
                        'class': self.classes[class_id],
                        'confidence': float(confidence),
                        'bbox': [x, y, w_box, h_box]
                    })
        
        return detections, image

def main():
    """Example usage"""
    detector = GloveDetector()
    
    # Example image path
    image_path = "dataset/glove/valid/images/your_image.jpg"
    if Path(image_path).exists():
        image = cv2.imread(image_path)
        detections, _ = detector.detect(image)
        
        print(f"Found {len(detections)} detections:")
        for det in detections:
            print(f"- {det['class']}: {det['confidence']:.2f}")

if __name__ == "__main__":
    main()
