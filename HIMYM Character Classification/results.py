import matplotlib.pyplot as plt

history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, val_loss_values, label="Validation Loss")
line2 = plt.plot(epochs, loss_values, label="Training Loss")
plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, val_acc_values, label="Validation Accuracy")
line2 = plt.plot(epochs, acc_values, label="Training Accuracy")
plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
import sys

np.set_printoptions(threshold=sys.maxsize)

y_pred = model.predict_classes(x_test)

cfReport = classification_report(np.argmax(y_test, axis=1), y_pred)
confMatrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred)

print(cfReport)
print(confMatrix)

plt.figure(figsize=(10,10))
plt.imshow(confMatrix, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([], [])
plt.yticks([], [])
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()

fig, axs = plt.subplots(5, 5)
fig.set_figheight(15)
fig.set_figwidth(15)

for i in range(5):
  for j in range(5):
    randomNum = np.random.randint(0, len(x_test))
    img2copy = x_test[randomNum]
    img = img2copy.copy()
    img[:, :, 0] = img2copy[:, :, 2]
    img[:, :, 2] = img2copy[:, :, 0]
    axs[i, j].imshow(img)
    img2copy = img2copy.reshape(1, 64, 64, 3)
    res = model.predict(img2copy)[0]
    axs[i, j].set_title(labels[np.where(res == np.amax(res))[0][0]])
    
for ax in axs.flat:
    ax.label_outer()

import os
import cv2
from keras.models import load_model

path = "C:/Users/ErenS/Desktop/faces/"

face_detection = cv2.CascadeClassifier(path + "haarcascade_frontalface_default.xml")
classifier = load_model(path + "himym.h5")
labels = ['Barney', 'Lilly', 'Robin', 'Marshall', 'Ted']

for file in os.listdir(path + "test"):
    img = cv2.imread(path + "test/" + file)
    faces = face_detection.detectMultiScale(img, scaleFactor=1.1, minNeighbors=8, minSize=(100, 100),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) > 0:
        for x, y, w, h in faces:
            roi = img[y:y + h, x:x + w]
            roi = roi.astype("float32")
            roi /= 255
            roi = cv2.resize(roi, (64, 64))
            roi = roi.reshape(1, 64, 64, 3)
            preds = classifier.predict(roi)[0]
            label = labels[preds.argmax()]
            cv2.rectangle(img, (x, y), (x + w, y + h), (10, 100, 80), 4)
            cv2.putText(img, label, (x - 40, y - 10), cv2.FONT_ITALIC, 0.45, (0, 0, 255), 1)

    cv2.imwrite(path + label + ".jpg", img)
