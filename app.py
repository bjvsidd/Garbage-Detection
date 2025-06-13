import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import tensorflow as tf

# Load models
detection_model = YOLO(r"D:\Downloads\waste_classification\Garbage_detection.pt")  # Path to YOLO model
classifier_model = tf.keras.models.load_model(r"D:\Downloads\waste_classification\waste_classifier_vgg16 (1).h5")  # Path to classifier
class_names = ['cardboard','Food Organic', 'glass','Metal','Miscellaneous Trash', 'paper','plastic','Textile Trash','Vegitation']  # Customize this

st.title("Garbage Detection & Classification App")

# Select input type
input_type = st.selectbox("Select Input Type", ['Image', 'Video'])

def classify_crop(crop):
    crop_resized = cv2.resize(crop, (256, 256)) / 255.0  # Normalize
    crop_expanded = np.expand_dims(crop_resized, axis=0)
    pred = classifier_model.predict(crop_expanded, verbose=0)
    return class_names[np.argmax(pred)]

def process_frame_with_classification(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detection_model.predict(rgb, conf=0.3, verbose=False)
    annotated_frame = frame.copy()

    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue
        label = classify_crop(cropped)

        # Draw bounding box + label
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return annotated_frame

# IMAGE INPUT
if input_type == 'Image':
    img_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if img_file:
        file_bytes = np.frombuffer(img_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

        with st.spinner("Processing..."):
            result_img = process_frame_with_classification(image)

        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Detected & Classified", use_container_width=True)
# VIDEO INPUT
else:
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 24

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        stframe = st.empty()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)

        count = 0
        with st.spinner("Processing video..."):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated = process_frame_with_classification(frame)
                out.write(annotated)
                count += 1
                progress.progress(min(count / total_frames, 1.0))

        cap.release()
        out.release()
        st.success("Video processing complete!")
        video_bytes = open(output_path, 'rb').read()
        st.video(video_bytes)
        st.download_button(
        label="📥 Download Processed Video",
        data=video_bytes,
        file_name="processed_video.avi",  # or .mp4 if converted
        mime="video/avi"  # or "video/mp4"
        )