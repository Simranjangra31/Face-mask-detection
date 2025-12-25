import tensorflow as tf
import os

model_path = "model/face_mask_detector.keras"
tflite_path = "model/face_mask_detector.tflite"

def convert_to_tflite():
    if not os.path.exists(model_path):
        print("Model not found!")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Quantization (Dynamic Range)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()

    print(f"Saving to {tflite_path}...")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    print("Optimization and Conversion Complete.")

if __name__ == "__main__":
    convert_to_tflite()


print("TFLite model saved")
