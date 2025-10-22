"""
Professional Hand Tracking System using OpenCV (No MediaPipe Required)
Provides moving bounding boxes for glove detection without dependency issues
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import tensorflow as tf
from tensorflow import keras


class OpenCVHandTracker:
    """Professional hand tracker using OpenCV methods only"""
    
    def __init__(self, glove_model_path: str, img_size: int = 640):
        """
        Initialize hand tracker with glove classification
        
        Args:
            glove_model_path: Path to trained glove classification model
            img_size: Input image size for glove model
        """
        # Load glove classification model
        self.glove_model = keras.models.load_model(glove_model_path)
        self.img_size = img_size
        self.classes = ['no_glove', 'glove']
        
        # OpenCV hand detection setup
        self.setup_hand_detection()
        
        # Tracking state
        self.previous_hands = []
        self.tracking_history = []
        
    def setup_hand_detection(self):
        """Setup OpenCV-based hand detection"""
        # Create background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50, history=500
        )
        
        # Morphological operations kernel
        self.kernel = np.ones((5, 5), np.uint8)
        
        # Improved skin color range (HSV) - more inclusive
        self.lower_skin = np.array([0, 10, 60], dtype=np.uint8)
        self.upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        
        # Additional skin color range for different lighting
        self.lower_skin2 = np.array([0, 20, 50], dtype=np.uint8)
        self.upper_skin2 = np.array([30, 255, 255], dtype=np.uint8)
        
    def detect_skin_regions(self, image: np.ndarray) -> np.ndarray:
        """Detect skin regions using color-based segmentation"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for skin color (primary range)
        skin_mask1 = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Create mask for skin color (secondary range)
        skin_mask2 = cv2.inRange(hsv, self.lower_skin2, self.upper_skin2)
        
        # Combine both masks
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        
        # Apply morphological operations to clean up the mask
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, self.kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Additional dilation to connect nearby skin regions
        skin_mask = cv2.dilate(skin_mask, self.kernel, iterations=2)
        
        return skin_mask
    
    def detect_hand_contours(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect hand contours and return bounding boxes"""
        # Get skin regions
        skin_mask = self.detect_skin_regions(image)
        
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hand_boxes = []
        image_area = image.shape[0] * image.shape[1]
        
        for contour in contours:
            # More flexible area filtering - based on image size
            area = cv2.contourArea(contour)
            min_area = max(1000, image_area * 0.001)  # At least 0.1% of image
            max_area = min(100000, image_area * 0.3)  # At most 30% of image
            
            if min_area < area < max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # More flexible aspect ratio (hands can be various shapes)
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 3.0:  # Much more flexible
                    # Additional check: ensure reasonable size
                    if w > 30 and h > 30:  # Minimum size
                        hand_boxes.append((x, y, w, h))
        
        return hand_boxes
    
    def detect_moving_objects(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect moving objects using background subtraction"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(image)
        
        # Apply morphological operations
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        moving_boxes = []
        image_area = image.shape[0] * image.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            min_area = max(1000, image_area * 0.0005)  # At least 0.05% of image
            max_area = min(50000, image_area * 0.2)   # At most 20% of image
            
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.2 < aspect_ratio < 5.0:  # More flexible aspect ratio
                    if w > 20 and h > 20:  # Minimum size
                        moving_boxes.append((x, y, w, h))
        
        return moving_boxes
    
    def combine_detections(self, skin_boxes: List, motion_boxes: List) -> List[Tuple[int, int, int, int]]:
        """Combine skin and motion detection results"""
        all_boxes = skin_boxes + motion_boxes
        
        # Remove overlapping boxes
        filtered_boxes = []
        for i, box1 in enumerate(all_boxes):
            x1, y1, w1, h1 = box1
            is_duplicate = False
            
            for j, box2 in enumerate(filtered_boxes):
                x2, y2, w2, h2 = box2
                
                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - overlap_area
                
                if union_area > 0 and overlap_area / union_area > 0.3:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_boxes.append(box1)
        
        return filtered_boxes
    
    def extract_hand_roi(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract hand region of interest from bounding box"""
        x, y, w, h = bbox
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        return image[y:y+h, x:x+w]
    
    def classify_hand(self, hand_roi: np.ndarray) -> Dict:
        """Classify hand as glove/no_glove"""
        if hand_roi.size == 0:
            return {'class': 'no_glove', 'confidence': 0.0, 'class_id': 0}
        
        # Preprocess for glove model
        processed = cv2.resize(hand_roi, (self.img_size, self.img_size))
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        processed = processed.astype('float32') / 255.0
        processed = np.expand_dims(processed, axis=0)
        
        # Predict
        predictions = self.glove_model.predict(processed, verbose=0)
        class_id = np.argmax(predictions[0])
        confidence = predictions[0][class_id]
        
        return {
            'class': self.classes[class_id],
            'confidence': float(confidence),
            'class_id': class_id
        }
    
    def smooth_tracking(self, current_hands: List[Dict], smoothing_factor: float = 0.7) -> List[Dict]:
        """Smooth hand tracking to reduce jitter"""
        if not self.previous_hands:
            self.previous_hands = current_hands.copy()
            return current_hands
        
        smoothed_hands = []
        for current_hand in current_hands:
            # Find closest previous hand
            min_distance = float('inf')
            closest_prev = None
            
            for prev_hand in self.previous_hands:
                distance = np.sqrt(
                    (current_hand['bbox'][0] - prev_hand['bbox'][0])**2 + 
                    (current_hand['bbox'][1] - prev_hand['bbox'][1])**2
                )
                if distance < min_distance:
                    min_distance = distance
                    closest_prev = prev_hand
            
            # Smooth position if close enough
            if closest_prev and min_distance < 100:  # 100 pixel threshold
                smoothed_bbox = [
                    int(smoothing_factor * closest_prev['bbox'][0] + (1 - smoothing_factor) * current_hand['bbox'][0]),
                    int(smoothing_factor * closest_prev['bbox'][1] + (1 - smoothing_factor) * current_hand['bbox'][1]),
                    int(smoothing_factor * closest_prev['bbox'][2] + (1 - smoothing_factor) * current_hand['bbox'][2]),
                    int(smoothing_factor * closest_prev['bbox'][3] + (1 - smoothing_factor) * current_hand['bbox'][3])
                ]
                current_hand['bbox'] = smoothed_bbox
            
            smoothed_hands.append(current_hand)
        
        self.previous_hands = smoothed_hands.copy()
        return smoothed_hands
    
    def draw_professional_tracking(self, image: np.ndarray, hands_data: List[Dict]) -> np.ndarray:
        """Draw professional tracking visualization"""
        result_image = image.copy()
        
        for hand_data in hands_data:
            bbox = hand_data['bbox']
            class_name = hand_data['class']
            confidence = hand_data['confidence']
            
            # Define colors
            if class_name == 'no_glove':
                color = (0, 0, 255)  # Red for no glove
                box_color = (0, 0, 255)
            else:
                color = (0, 255, 0)  # Green for glove
                box_color = (0, 255, 0)
            
            x, y, w, h = bbox
            
            # Draw main bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), box_color, 3)
            
            # Draw corner markers for professional look
            corner_length = 20
            corner_thickness = 3
            
            # Top-left corner
            cv2.line(result_image, (x, y), (x + corner_length, y), color, corner_thickness)
            cv2.line(result_image, (x, y), (x, y + corner_length), color, corner_thickness)
            
            # Top-right corner
            cv2.line(result_image, (x + w, y), (x + w - corner_length, y), color, corner_thickness)
            cv2.line(result_image, (x + w, y), (x + w, y + corner_length), color, corner_thickness)
            
            # Bottom-left corner
            cv2.line(result_image, (x, y + h), (x + corner_length, y + h), color, corner_thickness)
            cv2.line(result_image, (x, y + h), (x, y + h - corner_length), color, corner_thickness)
            
            # Bottom-right corner
            cv2.line(result_image, (x + w, y + h), (x + w - corner_length, y + h), color, corner_thickness)
            cv2.line(result_image, (x + w, y + h), (x + w, y + h - corner_length), color, corner_thickness)
            
            # Draw label with background
            label = f"{class_name}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            label_y = max(y - 10, text_height + 10)
            cv2.rectangle(result_image, (x, label_y - text_height - 10), 
                         (x + text_width + 10, label_y), (0, 0, 0), -1)
            cv2.rectangle(result_image, (x, label_y - text_height - 10), 
                         (x + text_width + 10, label_y), color, 2)
            
            # Draw label text
            cv2.putText(result_image, label, (x + 5, label_y - 5), font, font_scale, color, thickness)
        
        return result_image
    
    def detect_hands_fallback(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Fallback hand detection using edge detection and contour analysis"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hand_boxes = []
        image_area = image.shape[0] * image.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            min_area = max(2000, image_area * 0.002)  # At least 0.2% of image
            max_area = min(80000, image_area * 0.4)   # At most 40% of image
            
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Check for hand-like characteristics
                if 0.4 < aspect_ratio < 2.5 and w > 40 and h > 40:
                    # Additional check: convexity defects (fingers)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = area / hull_area
                        if 0.6 < solidity < 0.95:  # Hand-like solidity
                            hand_boxes.append((x, y, w, h))
        
        return hand_boxes

    def process_frame(self, image: np.ndarray, conf_threshold: float = 0.5) -> Tuple[np.ndarray, List[Dict]]:
        """Process single frame for hand tracking and glove detection"""
        
        # Detect hands using multiple methods
        skin_boxes = self.detect_hand_contours(image)
        motion_boxes = self.detect_moving_objects(image)
        fallback_boxes = self.detect_hands_fallback(image)
        
        # Combine all detections
        all_boxes = self.combine_detections(skin_boxes, motion_boxes)
        all_boxes = self.combine_detections(all_boxes, fallback_boxes)
        
        hands_data = []
        
        for bbox in all_boxes:
            # Extract hand region
            hand_roi = self.extract_hand_roi(image, bbox)
            
            # Classify glove/no_glove
            classification = self.classify_hand(hand_roi)
            
            if classification['confidence'] > conf_threshold:
                hands_data.append({
                    'bbox': bbox,
                    'class': classification['class'],
                    'confidence': classification['confidence'],
                    'class_id': classification['class_id']
                })
        
        # If no hands detected with high confidence, try with lower threshold
        if not hands_data:
            for bbox in all_boxes:
                hand_roi = self.extract_hand_roi(image, bbox)
                classification = self.classify_hand(hand_roi)
                
                # Use lower threshold for fallback
                if classification['confidence'] > max(0.1, conf_threshold * 0.5):
                    hands_data.append({
                        'bbox': bbox,
                        'class': classification['class'],
                        'confidence': classification['confidence'],
                        'class_id': classification['class_id']
                    })
        
        # Smooth tracking
        hands_data = self.smooth_tracking(hands_data)
        
        # Draw professional visualization
        processed_image = self.draw_professional_tracking(image, hands_data)
        
        return processed_image, hands_data
    
    def release(self):
        """Release resources"""
        pass


def main():
    """Example usage of OpenCVHandTracker"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Professional Hand Tracking with OpenCV (No MediaPipe)")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam)")
    parser.add_argument("--weights", type=str, required=True, help="Path to glove model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = OpenCVHandTracker(args.weights)
    
    # Open video source
    cap = cv2.VideoCapture(0 if args.source.isdigit() else args.source)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, hands_data = tracker.process_frame(frame, args.conf)
            
            # Display results
            cv2.imshow("Professional Hand Tracking (OpenCV)", processed_frame)
            
            # Print detection info
            for i, hand in enumerate(hands_data):
                print(f"Hand {i+1}: {hand['class']} (conf: {hand['confidence']:.2f})")
            
            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    finally:
        cap.release()
        tracker.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
