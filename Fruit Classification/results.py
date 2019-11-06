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

import matplotlib.pyplot as plt

history_dict = history.history

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
    img2copy = img2copy.reshape(1, 50, 50, 3)
    res = model.predict(img2copy)[0]
    axs[i, j].set_title(labels[np.where(res == np.amax(res))[0][0]])
    
for ax in axs.flat:
    ax.label_outer()
