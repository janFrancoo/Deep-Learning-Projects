import cv2

path = "C:/Users/ErenS/Desktop/himym/"
cap = cv2.VideoCapture(path + 'vid4.mp4')
frame_count = 0
face_detection = cv2.CascadeClassifier(path + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    frame_count += 1
    print(frame_count)
    count = 0

    faces = face_detection.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        for x, y, w, h in faces:
            roi = frame[y:y + h, x:x + w]
            roi = cv2.resize(roi, (64, 64))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (10, 100, 80), 4)
            file_name = path + "vid4/" + str(frame_count) + "_" + str(count) + ".jpg"
            count += 1
            cv2.imwrite(file_name, roi)

    cv2.imshow("Face Detector", frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
