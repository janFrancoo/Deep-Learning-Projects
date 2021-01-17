import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import config
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

INIT_LR = 1e-4
EPOCHS = 5
BATCH_SIZE = 16

raccoon_image_paths = [os.path.sep.join([config.POSITIVE_PATH, f]) for f in os.listdir(config.POSITIVE_PATH)]
no_raccoon_image_paths = [os.path.sep.join([config.NEGATIVE_PATH, f]) for f in os.listdir(config.NEGATIVE_PATH)]
image_paths = raccoon_image_paths + no_raccoon_image_paths

data = []
labels = []
for image_path in image_paths:
    data.append(preprocess_input(img_to_array(load_img(image_path, target_size=config.INPUT_DIMS))))
    labels.append(image_path.split(os.path.sep)[-2])

data = np.array(data, dtype="float32")
labels = np.array(labels)

label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = to_categorical(labels)

train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.2, stratify=labels)
aug = ImageDataGenerator(rotation_range=20, zoom_range=.15, width_shift_range=.2, height_shift_range=.2,
                         shear_range=.15, horizontal_flip=True, fill_mode="nearest")

base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 244, 3)))
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model)
model = Model(inputs=base_model.input, outputs=head_model)

opt = Adam(lr=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(aug.flow(train_x, train_y, batch_size=BATCH_SIZE), steps_per_epoch=len(train_x) // BATCH_SIZE,
              validation_data=(test_x, test_y), validation_steps=len(test_x) // BATCH_SIZE, epochs=EPOCHS)

pred_idxs = model.predict(test_x, batch_size=BATCH_SIZE)
pred_idxs = np.argmax(pred_idxs, axis=1)
print(classification_report(test_y.argmax(axis=1), pred_idxs, target_names=label_binarizer.classes_))

model.save(config.MODEL_PATH, save_format="h5")
f = open(config.ENCODER_PATH, "wb")
f.write(pickle.dumps(label_binarizer))
f.close()

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
