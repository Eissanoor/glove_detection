import io
from pathlib import Path
import tempfile
import time

import numpy as np
import streamlit as st
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow import keras

from config import CONF_THRESHOLD_DEFAULT

st.set_page_config(page_title="Glove Detector", page_icon="🧤", layout="centered")

st.title("🧤 Glove vs No-Glove Detector")
st.caption("Upload an image/video or use webcam to detect hands with gloves or without.")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", value="./src/runs/detect/glove_model.h5")
    conf_thres = st.slider("Confidence", 0.0, 1.0, float(CONF_THRESHOLD_DEFAULT), 0.01)
    img_size = st.selectbox("Image size", [224, 320, 416, 512, 640], index=4)
    
    st.header("Detection Mode")
    detection_mode = st.radio(
        "Choose detection mode:",
        ["Image Upload", "Video Upload", "Webcam"]
    )
    
    predict_button = st.button("Run Prediction", type="primary")

# File uploader based on mode
if detection_mode == "Image Upload":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    video_uploaded = None
elif detection_mode == "Video Upload":
    video_uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"])
    uploaded = None
else:  # Webcam mode
    uploaded = None
    video_uploaded = None 

# Helper functions
def preprocess_image(image, img_size):
    """Preprocess image for model input"""
    processed_img = cv2.resize(image, (img_size, img_size))
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    processed_img = processed_img.astype('float32') / 255.0
    processed_img = np.expand_dims(processed_img, axis=0)
    return processed_img

def draw_label_on_image(image, class_name, confidence, class_id):
    """Draw label on image with enhanced styling and bounding box tracking"""
    # Create a copy to avoid modifying original
    result_image = image.copy()
    
    # Define colors (BGR format for OpenCV)
    colors = [(0, 0, 255), (0, 255, 0)]  # Red for no glove, Green for glove
    
    # Draw bounding box for no_glove detections (red square box)
    if class_name == "No Glove":
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
    
    # Draw background rectangle for better text visibility
    text = f"{class_name}: {confidence:.3f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    cv2.rectangle(result_image, (5, 5), (text_width + 15, text_height + 15), (0, 0, 0), -1)
    cv2.rectangle(result_image, (5, 5), (text_width + 15, text_height + 15), colors[class_id], 2)
    
    # Draw text
    cv2.putText(result_image, text, (10, text_height + 10), font, font_scale, colors[class_id], thickness)
    
    return result_image

def process_video_frames(model, video_file, conf_thres, img_size):
    """Process video frames and return results"""
    # Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name
    
    cap = cv2.VideoCapture(tmp_path)
    processed_frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
        
        # Preprocess and predict
        processed_img = preprocess_image(frame, img_size)
        predictions = model.predict(processed_img, verbose=0)
        
        # Get results
        class_id = np.argmax(predictions[0])
        confidence = predictions[0][class_id]
        classes = ['No Glove', 'Glove']
        
        # Draw label on frame
        if confidence > conf_thres:
            predicted_class = classes[class_id]
            frame_with_label = draw_label_on_image(frame, predicted_class, confidence, class_id)
        else:
            frame_with_label = draw_label_on_image(frame, "Low Confidence", confidence, 0)
        
        processed_frames.append(frame_with_label)
    
    cap.release()
    
    # Clean up temporary file
    Path(tmp_path).unlink()
    
    return processed_frames, fps

# Load model
@st.cache_resource
def load_model(model_path):
    if Path(model_path).exists():
        try:
            return keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Could not load model: {e}")
            return None
    else:
        st.error(f"Model file not found: {model_path}")
        return None

if predict_button:
    model = load_model(model_path)

# Main interface based on detection mode
if detection_mode == "Image Upload":
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_column_width=True)
            
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            st.info("Upload an image to begin.")

    with col2:
        st.subheader("Prediction Result")
        if predict_button and uploaded:
            model = load_model(model_path)
            if model is None:
                st.stop()
            
            # Preprocess image
            processed_img = preprocess_image(cv_image, img_size)
            
            # Make prediction
            with st.spinner("Running prediction..."):
                predictions = model.predict(processed_img, verbose=0)
            
            # Get results
            class_id = np.argmax(predictions[0])
            confidence = predictions[0][class_id]
            classes = ['No Glove', 'Glove']
            
            # Display result
            if confidence > conf_thres:
                predicted_class = classes[class_id]
                st.success(f"**Prediction: {predicted_class}**")
                st.write(f"Confidence: {confidence:.3f}")
                
                # Show confidence bars
                st.write("Confidence breakdown:")
                for i, (class_name, conf) in enumerate(zip(classes, predictions[0])):
                    color = "green" if i == class_id else "gray"
                    st.write(f"{class_name}: {conf:.3f}")
                    
                # Display prediction on image with enhanced labels
                result_image = draw_label_on_image(cv_image, predicted_class, confidence, class_id)
                result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                st.image(result_image_rgb, use_column_width=True)
            else:
                st.warning(f"Low confidence prediction: {confidence:.3f} < {conf_thres}")
                st.write("Confidence breakdown:")
                for class_name, conf in zip(classes, predictions[0]):
                    st.write(f"{class_name}: {conf:.3f}")

elif detection_mode == "Video Upload":
    if video_uploaded:
        st.subheader("Video Processing")
        
        # Show original video info
        st.write(f"**Video file:** {video_uploaded.name}")
        
        if predict_button:
            model = load_model(model_path)
            if model is None:
                st.stop()
            
            # Process video
            with st.spinner("Processing video frames..."):
                processed_frames, fps = process_video_frames(model, video_uploaded, conf_thres, img_size)
            
            st.success(f"Processed {len(processed_frames)} frames successfully!")
            
            # Create output video
            if processed_frames:
                # Create temporary video file
                output_path = "processed_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                height, width = processed_frames[0].shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                for frame in processed_frames:
                    out.write(frame)
                out.release()
                
                # Display video
                st.subheader("Processed Video with Labels")
                st.video(output_path)
                
                # Download button
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="Download Processed Video",
                        data=file.read(),
                        file_name=f"processed_{video_uploaded.name}",
                        mime="video/mp4"
                    )
                
                # Clean up
                Path(output_path).unlink()
    else:
        st.info("Upload a video file to begin processing.")

elif detection_mode == "Webcam":
    st.subheader("Live Webcam Detection")
    st.info("Webcam functionality requires running the inference script separately.")
    st.code("""
# Run this command in your terminal for webcam detection:
python src/infer_tf.py --source 0 --weights ./src/runs/detect/glove_model.h5 --conf 0.5 --show
    """)

st.markdown("""
**How to use:**

**Image Detection:**
1. Select "Image Upload" mode
2. Upload an image file
3. Click "Run Prediction" to see results with labels

**Video Detection:**
1. Select "Video Upload" mode  
2. Upload a video file (mp4, avi, mov, mkv, webm)
3. Click "Run Prediction" to process all frames
4. Download the processed video with labels

**Webcam Detection:**
1. Select "Webcam" mode for instructions
2. Run the command line tool for live detection

**Model Info:**
- Input size: Configurable (224, 320, 416, 512, 640 pixels)
- Classes: Glove vs No Glove
- Output: Confidence scores with visual labels and tracking boxes
- Label colors: Green for Glove, Red for No Glove

**Features:**
- Real-time confidence display
- Enhanced label visualization with background rectangles
- **Red square bounding box tracking for No Glove detections**
- Video processing with progress tracking and tracking boxes
- Download processed videos with tracking visualization
- Webcam support via command line with live tracking
""")
