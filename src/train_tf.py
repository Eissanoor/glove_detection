"""
Alternative training script using TensorFlow/Keras instead of PyTorch
This avoids the PyTorch DLL issues entirely
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


def load_dataset(data_yaml_path, img_size=640):
    """Load dataset from YAML config"""
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Load images and labels
    train_images = []
    train_labels = []
    
    # Load training data - fix the path resolution
    yaml_dir = Path(data_yaml_path).parent
    # The data.yaml has '../train/images' but we need 'train/images'
    train_img_dir = yaml_dir / 'train' / 'images'
    train_label_dir = yaml_dir / 'train' / 'labels'
    
    for img_path in train_img_dir.glob('*.jpg'):
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        train_images.append(img)
        
        # Load label
        label_path = train_label_dir / (img_path.stem + '.txt')
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                # Simple binary classification: glove (1) or no_glove (0)
                has_glove = any(line.strip().startswith('0') for line in lines)  # class 0 = glove
                train_labels.append(1 if has_glove else 0)
        else:
            train_labels.append(0)  # default to no_glove
    
    return np.array(train_images), np.array(train_labels)


def create_model(input_shape, num_classes=2):
    """Create a simple CNN model for glove detection"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN for glove detection using TensorFlow")
    parser.add_argument("--data", type=str, default=str(DATA_YAML), help="path to data.yaml")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="number of epochs")
    parser.add_argument("--img", type=int, default=DEFAULT_IMAGE_SIZE, help="image size")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    
    print("Loading dataset...")
    images, labels = load_dataset(args.data, args.img)
    print(f"Loaded {len(images)} images")
    
    # Normalize images
    images = images.astype('float32') / 255.0
    
    # Create model
    model = create_model((args.img, args.img, 3))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model summary:")
    model.summary()
    
    # Train model
    print("Training model...")
    history = model.fit(
        images, labels,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_split=0.2,
        verbose=1
    )
    
    # Save model
    model_path = "runs/detect/glove_model.h5"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Print final accuracy
    final_acc = history.history['accuracy'][-1]
    print(f"Final training accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()
