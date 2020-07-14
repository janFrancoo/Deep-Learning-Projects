import cv2
import argparse
from tensorflow.keras.models import load_model

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image path")
ap.add_argument("-m", "--model", required=True, help="Trained model path")
ap.add_argument("-c", "--haar_cascade", required=True, help="Path of Haar Cascade")
args = vars(ap.parse_args())

model = load_model(args["model"])
img = cv2.imread(args["image"])
face_cascade = cv2.CascadeClassifier(args["haar_cascade"])

grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_rects = face_cascade.detectMultiScale(grayscale, scaleFactor=1.3, minNeighbors=5)

for x, y, w, h in face_rects:
    roi_face = img[y:y+h, x:x+w]
    resized = cv2.resize(roi_face, (128, 128))
    resized = resized.astype("float32")
    resized /= 255.0
    resized = resized.reshape(1, 128, 128, 3)
    res = model.predict(resized) > 0.5

    if not res:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, 'No mask!', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, 'Wearing mask!', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("final", img)
    cv2.waitKey(0)
