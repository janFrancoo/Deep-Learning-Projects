import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

prediction_queue = deque()
classes = ['Cyclone', 'Earthquake', 'Flood', 'Wildfire']
model = load_model("C:/Users/PC/Documents/Data/disaster/best.h5")
vid = cv2.VideoCapture("C:/Users/PC/Documents/Data/disaster/wildfire.mp4")

while True:
    (grabbed, frame) = vid.read()

    if not grabbed:
        break

    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype("float32")

    prediction = model.predict(np.expand_dims(frame, axis=0))[0]
    prediction_queue.append(prediction)

    results = np.array(prediction_queue).mean(axis=0)
    label = classes[np.argmax(results)]
    cv2.putText(output, label + "!", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

vid.release()
