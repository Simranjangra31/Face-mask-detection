import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# Constants
IMG_SIZE = 224
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]
COLORS = [(0, 255, 0), (0, 0, 255), (255, 255, 0)] # BGR
# Green for Mask, Red for No Mask, Yellow for Incorrect

@st.cache_resource
def load_face_mask_model():
    return tf.keras.models.load_model("model/face_mask_detector.keras")

st.title("Face Mask Detection App")
st.write("Upload an image to detect if people are wearing masks correctly.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and Preprocess
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1) # BGR
    original_image = image.copy()
    h, w, _ = image.shape
    
    # Resize for model
    image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image_normalized = image_resized / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    # Predict
    if st.button("Detect Mask"):
        try:
            model = load_face_mask_model()
            predictions = model.predict(image_batch)
            
            # Unpack predictions
            class_probs = predictions[0][0]
            bbox = predictions[1][0]
            
            class_idx = np.argmax(class_probs)
            confidence = class_probs[class_idx]
            label = CLASSES[class_idx]
            
            # Draw Bounding Box
            xmin, ymin, xmax, ymax = bbox
            xmin = int(xmin * w)
            ymin = int(ymin * h)
            xmax = int(xmax * w)
            ymax = int(ymax * h)
            
            color = COLORS[class_idx]
            
            cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), color, 3)
            
            label_text = f"{label} ({confidence*100:.1f}%)"
            cv2.putText(original_image, label_text, (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Display Result
            st.image(original_image, channels="BGR", use_column_width=True)
            st.success(f"Detected: **{label}** with {confidence*100:.1f}% confidence.")
            
        except Exception as e:
            st.error(f"Error occurred: {e}")
            st.write("Make sure 'model/face_mask_detector.keras' exists.")

