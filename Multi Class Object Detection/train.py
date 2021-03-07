import cv2
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import img_to_array, load_img

data = []
labels = []
boxes = []
image_paths = []

csv_files = ["airplane.csv", "face.csv", "motorcycle.csv"]
for csv_file in csv_files:
    csv_path = "dataset/annotations/" + csv_file
    rows = open(csv_path).read().strip().split("\n")
    for row in rows:
        row = row.split(",")
        file_name, start_x, start_y, end_x, end_y, label = row
        image_path = "dataset/images/" + label + "/" + file_name

        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        start_x = float(start_x) / w
        start_y = float(start_y) / h
        end_x = float(end_x) / w
        end_y = float(end_y) / h
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)

        data.append(image)
        labels.append(label)
        boxes.append([start_x, start_y, end_x, end_y])
        image_paths.append(image_path)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
boxes = np.array(boxes, dtype="float32")
image_paths = np.array(image_paths)

label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)

train_images, test_images, train_labels, test_labels, train_boxes, test_boxes, train_image_paths, test_image_paths = \
    train_test_split(data, labels, boxes, image_paths, test_size=.2)

with open("test_images.txt", "w") as file:
    file.write("\n".join(test_image_paths))

vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
vgg.trainable = False
flatten = vgg.output
flatten = Flatten()(flatten)

box_head = Dense(128, activation="relu")(flatten)
box_head = Dense(64, activation="relu")(box_head)
box_head = Dense(32, activation="relu")(box_head)
box_head = Dense(4, activation="sigmoid", name="bounding_box")(box_head)

softmax_head = Dense(512, activation="relu")(flatten)
softmax_head = Dropout(0.5)(softmax_head)
softmax_head = Dense(512, activation="relu")(softmax_head)
softmax_head = Dropout(0.5)(softmax_head)
softmax_head = Dense(3, activation="softmax", name="class_label")(softmax_head)

model = Model(inputs=vgg.input, outputs=(box_head, softmax_head))

model.compile(
    metrics=["accuracy"],
    optimizer=Adam(lr=1e-4),
    loss_weights={"class_label": 1.0, "bounding_box": 1.0},
    loss={"class_label": "categorical_crossentropy", "bounding_box": "mean_squared_error"}
)

history = model.fit(
    train_images, {"class_label": train_labels, "bounding_box": train_boxes},
    validation_data=(test_images, {"class_label": test_labels, "bounding_box": test_boxes}),
    batch_size=32, epochs=20, verbose=1
)

model.save("detector.h5", save_format="h5")

with open("label_binarizer.pickle", "wb") as file:
    file.write(pickle.dumps(label_binarizer))
