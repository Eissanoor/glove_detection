import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from config import CONF_THRESHOLD_DEFAULT, IOU_THRESHOLD_DEFAULT

st.set_page_config(page_title="Glove Detector", page_icon="🧤", layout="centered")

st.title("🧤 Glove vs No-Glove Detector")
st.caption("Upload an image and run YOLOv8 to detect hands with gloves or without.")

with st.sidebar:
	st.header("Settings")
	weights_path = st.text_input("Weights path", value="runs/detect/train/weights/best.pt")
	conf_thres = st.slider("Confidence", 0.0, 1.0, float(CONF_THRESHOLD_DEFAULT), 0.01)
	iou_thres = st.slider("IoU (NMS)", 0.0, 1.0, float(IOU_THRESHOLD_DEFAULT), 0.01)
	predict_button = st.button("Run Prediction", type="primary")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]) 

if "model" not in st.session_state and Path(weights_path).exists():
	try:
		st.session_state.model = YOLO(weights_path)
	except Exception as e:
		st.warning(f"Could not load model: {e}")

col1, col2 = st.columns(2)
with col1:
	st.subheader("Input")
	if uploaded:
		image = Image.open(uploaded).convert("RGB")
		st.image(image, use_column_width=True)
	else:
		st.info("Upload an image to begin.")

with col2:
	st.subheader("Prediction")
	if predict_button and uploaded:
		if "model" not in st.session_state or not isinstance(st.session_state.model, YOLO):
			try:
				st.session_state.model = YOLO(weights_path)
			except Exception as e:
				st.error(f"Failed to load model: {e}")
				st.stop()

		# Run prediction
		result = st.session_state.model.predict(source=np.array(image), conf=conf_thres, iou=iou_thres, verbose=False)[0]
		pred_image = Image.fromarray(result.plot())
		st.image(pred_image, use_column_width=True)

		# Show class counts
		if result.boxes is not None and len(result.boxes) > 0:
			classes = result.names
			class_ids = result.boxes.cls.cpu().numpy().astype(int)
			counts = {}
			for cid in class_ids:
				label = classes.get(cid, str(cid)) if isinstance(classes, dict) else str(cid)
				counts[label] = counts.get(label, 0) + 1
			st.write("Detections:")
			for label, count in counts.items():
				st.write(f"- {label}: {count}")
		else:
			st.write("No detections.")

st.markdown("""
Tips:
- Train first via `python -m src.train --data dataset/glove/data.yaml`.
- After training, set the weights path to your `best.pt`.
""")
