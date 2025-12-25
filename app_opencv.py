import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("model/face_mask_detector")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224,224)) / 255.0
    box, _ = model.predict(np.expand_dims(img, 0))

    h, w, _ = frame.shape
    xmin, ymin, xmax, ymax = (box[0] * [w,h,w,h]).astype(int)

    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
    cv2.imshow("Mask Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
