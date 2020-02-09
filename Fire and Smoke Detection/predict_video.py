import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

prediction_queue = deque()
model = load_model("C:/Users/PC/Documents/Data/fire-smoke/best.h5")
vid = cv2.VideoCapture("C:/Users/PC/Documents/Data/fire-smoke/crash.mp4")

while True:
    (grabbed, frame) = vid.read()

    if not grabbed:
        break

    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128))
    frame = frame.astype("float32")
    frame /= 255

    prediction = model.predict_classes(np.expand_dims(frame, axis=0))[0]
    prediction_queue.append(prediction)

    result = np.array(prediction_queue).mean(axis=0)
    if result[0] > 0.4:
        cv2.putText(output, "Warning! Fire!", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 7)

    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

vid.release()
