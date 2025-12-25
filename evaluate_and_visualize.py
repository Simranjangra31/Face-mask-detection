import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, average_precision_score
import os

IMG_SIZE = 224
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]
IOU_THRESHOLD = 0.5

def calculate_iou(box1, box2):
    # box: [xmin, ymin, xmax, ymax]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0

def evaluate():
    print("Loading data and model...")
    data = np.load("data/dataset.npz")
    X_test = data["X_test"]
    yb_test = data["yb_test"]
    yl_test = data["yl_test"]
    
    model = tf.keras.models.load_model("model/face_mask_detector.keras")
    
    print("Predicting on test set...")
    pred_probs, pred_boxes = model.predict(X_test, verbose=1)
    
    # Classification Metrics
    y_true_cls = np.argmax(yl_test, axis=1)
    y_pred_cls = np.argmax(pred_probs, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true_cls, y_pred_cls, target_names=CLASSES))
    
    # Bounding Box Metrics (mAP approx via IoU Accuracy)
    correct_boxes = 0
    total_boxes = len(yb_test)
    ious = []
    
    for i in range(total_boxes):
        iou = calculate_iou(yb_test[i], pred_boxes[i])
        ious.append(iou)
        if iou >= IOU_THRESHOLD:
            correct_boxes += 1
            
    print(f"\nMean IoU: {np.mean(ious):.4f}")
    print(f"Accuracy @ IoU=0.5: {correct_boxes / total_boxes:.4f}")

    # Visualization
    visualize(X_test, yb_test, yl_test, pred_boxes, y_pred_cls, model_path="model/face_mask_detector.keras")

                      (int(p_box[0]*w), int(p_box[1]*h)), 
                      (int(p_box[2]*w), int(p_box[3]*h)), 
                      (0, 0, 255), 2)
        
        # Label
        label_text = f"True: {CLASSES[np.argmax(y_label[i])]} | Pred: {CLASSES[pred_label[i]]}"
        cv2.putText(img, label_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save
        # Convert RGB to BGR for OpenCV saving
        img_bgr = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"predictions/pred_{i}.png", img_bgr * 255)
        
    print(f"\nSaved {num_samples} visualized predictions to 'predictions/' directory.")

if __name__ == "__main__":
    if not os.path.exists("model/face_mask_detector.keras"):
        print("Model not found. Please train first.")
    else:
        evaluate()


