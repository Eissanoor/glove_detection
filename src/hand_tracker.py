"""
Professional Hand Tracking System for Glove Detection
Combines MediaPipe hand tracking with glove classification
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict
import tensorflow as tf
from tensorflow import keras


class ProfessionalHandTracker:
    def __init__(self, glove_model_path: str, img_size: int = 640):
        """
        Initialize professional hand tracker with glove classification
        
        Args:
            glove_model_path: Path to trained glove classification model
            img_size: Input image size for glove model
        """
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Load glove classification model
        self.glove_model = keras.models.load_model(glove_model_path)
        self.img_size = img_size
        self.classes = ['no_glove', 'glove']
        
        # Tracking state
        self.previous_hands = []
        self.tracking_history = []
        
    def extract_hand_roi(self, image: np.ndarray, hand_landmarks) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract hand region of interest from landmarks
        
        Args:
            image: Input image
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            cropped_hand: Hand region image
            bbox: Bounding box (x, y, w, h)
        """
        h, w = image.shape[:2]
        
        # Get hand bounding box from landmarks
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        
        x_min = int(min(x_coords) * w)
        y_min = int(min(y_coords) * h)
        x_max = int(max(x_coords) * w)
        y_max = int(max(y_coords) * h)
        
        # Add padding around hand
        padding = 30
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Extract hand region
        hand_roi = image[y_min:y_max, x_min:x_max]
        
        return hand_roi, (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def classify_hand(self, hand_roi: np.ndarray) -> Dict:
        """
        Classify hand as glove/no_glove
        
        Args:
            hand_roi: Hand region image
            
        Returns:
            Classification result with confidence
        """
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
        """
        Smooth hand tracking to reduce jitter
        
        Args:
            current_hands: Current frame hand detections
            smoothing_factor: Smoothing factor (0-1)
            
        Returns:
            Smoothed hand positions
        """
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
        """
        Draw professional tracking visualization
        
        Args:
            image: Input image
            hands_data: List of hand detection data
            
        Returns:
            Image with professional tracking visualization
        """
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
    
    def process_frame(self, image: np.ndarray, conf_threshold: float = 0.5) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process single frame for hand tracking and glove detection
        
        Args:
            image: Input image
            conf_threshold: Confidence threshold for glove detection
            
        Returns:
            processed_image: Image with tracking visualization
            hands_data: List of detected hands with classifications
        """
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        results = self.hands.process(rgb_image)
        
        hands_data = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract hand region
                hand_roi, bbox = self.extract_hand_roi(image, hand_landmarks)
                
                # Classify glove/no_glove
                classification = self.classify_hand(hand_roi)
                
                if classification['confidence'] > conf_threshold:
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
        """Release MediaPipe resources"""
        if hasattr(self, 'hands'):
            self.hands.close()


def main():
    """Example usage of ProfessionalHandTracker"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Professional Hand Tracking with Glove Detection")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam)")
    parser.add_argument("--weights", type=str, required=True, help="Path to glove model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = ProfessionalHandTracker(args.weights)
    
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
            cv2.imshow("Professional Hand Tracking", processed_frame)
            
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
