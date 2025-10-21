"""
Advanced training script to enhance the training to support bounding box regression alongside classification
"""

import argparse
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import yaml

from config import DATA_YAML, DEFAULT_EPOCHS, DEFAULT_IMAGE_SIZE, ensure_dirs


class GloveDetectionModel:
    """Advanced model that can do both classification and bounding box regression"""
    
    def __init__(self, input_shape, num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self):
        """Build model with shared backbone and separate heads for classification and bbox regression"""
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='image_input')
        
        # Shared backbone (feature extractor)
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Classification head
        classification = layers.Dense(128, activation='relu')(x)
        classification = layers.Dropout(0.5)(classification)
        classification_output = layers.Dense(self.num_classes, activation='softmax', name='classification')(classification)
        
        # Bounding box regression head
        bbox = layers.Dense(128, activation='relu')(x)
        bbox = layers.Dropout(0.5)(bbox)
        bbox_output = layers.Dense(4, activation='sigmoid', name='bbox')(bbox)  # x, y, w, h normalized
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=[classification_output, bbox_output])
        
        return model
    
    def compile_model(self):
        """Compile model with appropriate losses and metrics"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'classification': 'sparse_categorical_crossentropy',
                'bbox': 'mse'
            },
            loss_weights={
                'classification': 1.0,
                'bbox': 0.5  # Lower weight for bbox to balance losses
            },
            metrics={
                'classification': ['accuracy'],
                'bbox': ['mae']
            }
        )


def load_advanced_dataset(data_yaml_path, img_size=640):
    """Load dataset with both classification labels and bounding boxes"""
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    images = []
    classification_labels = []
    bbox_labels = []
    
    yaml_dir = Path(data_yaml_path).parent
    train_img_dir = yaml_dir / 'train' / 'images'
    train_label_dir = yaml_dir / 'train' / 'labels'
    
    for img_path in train_img_dir.glob('*.jpg'):
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        
        # Load labels
        label_path = train_label_dir / (img_path.stem + '.txt')
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            # Parse YOLO format: class x_center y_center width height (normalized)
            if lines:
                line = lines[0].strip().split()
                class_id = int(line[0])
                x_center = float(line[1])
                y_center = float(line[2])
                width = float(line[3])
                height = float(line[4])
                
                classification_labels.append(class_id)
                bbox_labels.append([x_center, y_center, width, height])
            else:
                classification_labels.append(0)  # Default to no_glove
                bbox_labels.append([0.5, 0.5, 0.1, 0.1])  # Default bbox
        else:
            classification_labels.append(0)  # Default to no_glove
            bbox_labels.append([0.5, 0.5, 0.1, 0.1])  # Default bbox
    
    return (np.array(images), 
            np.array(classification_labels), 
            np.array(bbox_labels))


def parse_args():
    parser = argparse.ArgumentParser(description="Train advanced glove detection model")
    parser.add_argument("--data", type=str, default=str(DATA_YAML), help="path to data.yaml")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="number of epochs")
    parser.add_argument("--img", type=int, default=DEFAULT_IMAGE_SIZE, help="image size")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--mode", type=str, default="both", choices=["classification", "bbox", "both"],
                       help="Training mode: classification only, bbox only, or both")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    
    print("Loading advanced dataset...")
    images, class_labels, bbox_labels = load_advanced_dataset(args.data, args.img)
    print(f"Loaded {len(images)} images")
    print(f"Classification labels shape: {class_labels.shape}")
    print(f"Bbox labels shape: {bbox_labels.shape}")
    
    # Normalize images
    images = images.astype('float32') / 255.0
    
    # Create and compile model
    glove_model = GloveDetectionModel((args.img, args.img, 3))
    glove_model.compile_model()
    
    print("Model summary:")
    glove_model.model.summary()
    
    # Prepare training data based on mode
    if args.mode == "classification":
        train_data = images
        train_labels = class_labels
        glove_model.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    elif args.mode == "bbox":
        train_data = images
        train_labels = bbox_labels
        glove_model.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
    else:  # both
        train_data = images
        train_labels = {
            'classification': class_labels,
            'bbox': bbox_labels
        }
        glove_model.compile_model()
    
    # Train model
    print(f"Training model in {args.mode} mode...")
    history = glove_model.model.fit(
        train_data, train_labels,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_split=0.2,
        verbose=1
    )
    
    # Save model
    model_path = "runs/detect/advanced_glove_model.h5"
    glove_model.model.save(model_path)
    print(f"Advanced model saved to {model_path}")
    
    # Print final metrics
    if args.mode == "both":
        final_class_acc = history.history['classification_accuracy'][-1]
        final_bbox_mae = history.history['bbox_mae'][-1]
        print(f"Final classification accuracy: {final_class_acc:.4f}")
        print(f"Final bbox MAE: {final_bbox_mae:.4f}")
    else:
        if args.mode == "classification":
            final_acc = history.history['accuracy'][-1]
            print(f"Final classification accuracy: {final_acc:.4f}")
        else:
            final_mae = history.history['mae'][-1]
            print(f"Final bbox MAE: {final_mae:.4f}")


if __name__ == "__main__":
    main()
