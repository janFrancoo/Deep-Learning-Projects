import os
import scipy
import imageio
import numpy as np
from keras import backend as K
from PyQt5 import QtCore, QtWidgets
from keras.applications import vgg16
from keras.preprocessing import image
from PyQt5.QtWidgets import QFileDialog
from keras.applications import inception_v3


class Window(QtWidgets.QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(380, 190)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(270, 130, 101, 31))
        self.pushButton.setObjectName("pushButton")
        self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton.setGeometry(QtCore.QRect(10, 10, 81, 17))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_2.setGeometry(QtCore.QRect(10, 40, 82, 17))
        self.radioButton_2.setObjectName("radioButton_2")
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox.setGeometry(QtCore.QRect(210, 10, 51, 22))
        self.doubleSpinBox.setDecimals(2)
        self.doubleSpinBox.setMaximum(10.0)
        self.doubleSpinBox.setSingleStep(0.01)
        self.doubleSpinBox.setProperty("value", 0.08)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_2.setGeometry(QtCore.QRect(210, 40, 51, 22))
        self.doubleSpinBox_2.setDecimals(2)
        self.doubleSpinBox_2.setMaximum(10.0)
        self.doubleSpinBox_2.setSingleStep(0.01)
        self.doubleSpinBox_2.setProperty("value", 0.3)
        self.doubleSpinBox_2.setObjectName("doubleSpinBox_2")
        self.doubleSpinBox_3 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_3.setGeometry(QtCore.QRect(210, 70, 51, 22))
        self.doubleSpinBox_3.setDecimals(2)
        self.doubleSpinBox_3.setMaximum(10.0)
        self.doubleSpinBox_3.setSingleStep(0.01)
        self.doubleSpinBox_3.setProperty("value", 0.4)
        self.doubleSpinBox_3.setObjectName("doubleSpinBox_3")
        self.doubleSpinBox_5 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_5.setGeometry(QtCore.QRect(310, 10, 51, 22))
        self.doubleSpinBox_5.setDecimals(2)
        self.doubleSpinBox_5.setMaximum(10.0)
        self.doubleSpinBox_5.setSingleStep(0.01)
        self.doubleSpinBox_5.setProperty("value", 0.9)
        self.doubleSpinBox_5.setObjectName("doubleSpinBox_5")
        self.doubleSpinBox_6 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_6.setGeometry(QtCore.QRect(310, 40, 51, 22))
        self.doubleSpinBox_6.setDecimals(2)
        self.doubleSpinBox_6.setMaximum(10.0)
        self.doubleSpinBox_6.setSingleStep(0.01)
        self.doubleSpinBox_6.setProperty("value", 1.2)
        self.doubleSpinBox_6.setObjectName("doubleSpinBox_6")
        self.doubleSpinBox_7 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_7.setGeometry(QtCore.QRect(310, 70, 51, 22))
        self.doubleSpinBox_7.setDecimals(2)
        self.doubleSpinBox_7.setMaximum(10.0)
        self.doubleSpinBox_7.setSingleStep(0.01)
        self.doubleSpinBox_7.setProperty("value", 1.8)
        self.doubleSpinBox_7.setObjectName("doubleSpinBox_7")
        self.doubleSpinBox_8 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_8.setGeometry(QtCore.QRect(310, 100, 51, 22))
        self.doubleSpinBox_8.setDecimals(2)
        self.doubleSpinBox_8.setMaximum(10.0)
        self.doubleSpinBox_8.setSingleStep(0.01)
        self.doubleSpinBox_8.setProperty("value", 0.9)
        self.doubleSpinBox_8.setObjectName("doubleSpinBox_8")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 70, 121, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(130, 10, 71, 20))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(130, 40, 71, 20))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(130, 70, 71, 20))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(270, 10, 41, 20))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(270, 40, 41, 20))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(270, 70, 41, 20))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(270, 100, 41, 20))
        self.label_8.setObjectName("label_8")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(10, 110, 165, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(180, 110, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(180, 140, 75, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(10, 140, 165, 20))
        self.lineEdit_2.setObjectName("lineEdit_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 379, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Deep Dream - JanFranco"))
        self.pushButton.setText(_translate("MainWindow", "START !"))
        self.radioButton.setText(_translate("MainWindow", "VGG19"))
        self.radioButton_2.setText(_translate("MainWindow", "Inception v3"))
        self.label.setText(_translate("MainWindow", ""))
        self.label_2.setText(_translate("MainWindow", "block4_conv1"))
        self.label_3.setText(_translate("MainWindow", "block4_conv2"))
        self.label_4.setText(_translate("MainWindow", "block4_conv3"))
        self.label_5.setText(_translate("MainWindow", "mixed2"))
        self.label_6.setText(_translate("MainWindow", "mixed3"))
        self.label_7.setText(_translate("MainWindow", "mixed4"))
        self.label_8.setText(_translate("MainWindow", "mixed5"))
        self.pushButton_2.setText(_translate("MainWindow", "Image Path"))
        self.pushButton_3.setText(_translate("MainWindow", "Save Path"))
        self.pushButton.clicked.connect(self.start)
        self.pushButton_2.clicked.connect(self.select_file)
        self.pushButton_3.clicked.connect(self.save_file)
        self.radioButton.clicked.connect(self.setVGG)
        self.radioButton_2.clicked.connect(self.setInception)
        self.inputPath = ""
        self.outputPath = ""

    def setVGG(self):
        self.label.setText("")
        self.doubleSpinBox.setEnabled(True)
        self.doubleSpinBox_2.setEnabled(True)
        self.doubleSpinBox_3.setEnabled(True)
        self.doubleSpinBox_5.setEnabled(False)
        self.doubleSpinBox_6.setEnabled(False)
        self.doubleSpinBox_7.setEnabled(False)
        self.doubleSpinBox_8.setEnabled(False)

    def setInception(self):
        self.label.setText("")
        self.doubleSpinBox.setEnabled(False)
        self.doubleSpinBox_2.setEnabled(False)
        self.doubleSpinBox_3.setEnabled(False)
        self.doubleSpinBox_5.setEnabled(True)
        self.doubleSpinBox_6.setEnabled(True)
        self.doubleSpinBox_7.setEnabled(True)
        self.doubleSpinBox_8.setEnabled(True)

    def start(self):
        if self.radioButton.isChecked() == False and self.radioButton_2.isChecked() == False:
            self.label.setText("Error! No model is\nselected!")
        elif self.inputPath == "" or self.outputPath == "":
            self.label.setText("Error! Paths are not\nselected!")
        else:
            self.label.setText("Working...")
            K.set_learning_phase(0)
            if self.radioButton.isChecked():
                print("VGG")
                model = vgg16.VGG16(weights='imagenet', include_top=False)
                dream = model.input
                settings = {
                    'features': {
                        'block4_conv1': float(self.doubleSpinBox.text().replace(",", ".")),
                        'block4_conv2': float(self.doubleSpinBox_2.text().replace(",", ".")),
                        'block4_conv3': float(self.doubleSpinBox_3.text().replace(",", "."))
                    },
                }
            else:
                print("INCEPTION")
                model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
                dream = model.input
                settings = {
                    'features': {
                        'mixed2': float(self.doubleSpinBox_5.text()),
                        'mixed3': float(self.doubleSpinBox_6.text()),
                        'mixed4': float(self.doubleSpinBox_7.text()),
                        'mixed5': float(self.doubleSpinBox_8.text())
                    },
                }
            layer_dict = dict([(layer.name, layer) for layer in model.layers])

            for layer_name in settings['features']:
                coeff = settings['features'][layer_name]
                x = layer_dict[layer_name].output
                scaling = K.prod(K.cast(K.shape(x), 'float32'))
                loss = coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

            grads = K.gradients(loss, dream)[0]
            grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())

            outputs = [loss, grads]
            fetch_loss_and_grads = K.function([dream], outputs)

            def eval_loss_and_grads(x):
                outs = fetch_loss_and_grads([x])
                loss_value = outs[0]
                grad_values = outs[1]
                return loss_value, grad_values

            def gradient_ascent(x, iterations, step, max_loss=None):
                for i in range(iterations):
                    loss_value, grad_values = eval_loss_and_grads(x)
                    if max_loss is not None and loss_value > max_loss:
                        break
                    self.label.setText('...Loss value at ' + str(i) + ':\n' + str(loss_value))
                    x += step * grad_values
                return x

            def resize_img(img, size):
                img = np.copy(img)
                factors = (1,
                           float(size[0]) / img.shape[1],
                           float(size[1]) / img.shape[2], 1)
                return scipy.ndimage.zoom(img, factors, order=1)

            def save_img(img, fname):
                pil_img = deprocess_image(np.copy(img))
                imageio.imwrite(fname, pil_img)

            def preprocess_image(image_path):
                img = image.load_img(image_path)
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = inception_v3.preprocess_input(img)
                return img

            def deprocess_image(x):
                x = x.reshape((x.shape[1], x.shape[2], 3))
                x /= 2.
                x += 0.5
                x *= 255.
                x = np.clip(x, 0, 255).astype('uint8')
                return x

            step = 0.01
            num_octave = 3
            octave_scale = 1.2
            iterations = 20
            max_loss = 20.0

            base_image_path = self.lineEdit.text()

            img = preprocess_image(base_image_path)

            original_shape = img.shape[1:3]
            successive_shapes = [original_shape]
            for i in range(1, num_octave):
                shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
                successive_shapes.append(shape)

            successive_shapes = successive_shapes[::-1]

            original_img = np.copy(img)
            shrunk_original_img = resize_img(img, successive_shapes[0])

            for shape in successive_shapes:
                img = resize_img(img, shape)
                img = gradient_ascent(img, iterations=iterations, step=step, max_loss=max_loss)
                upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
                same_size_original = resize_img(original_img, shape)
                lost_detail = same_size_original - upscaled_shrunk_original_img

                img += lost_detail
                shrunk_original_img = resize_img(original_img, shape)

            save_img(img, self.lineEdit_2.text())


    def select_file(self):
        filter = "Images(*.jpg *.jpeg *.png)"
        fileName = QFileDialog.getOpenFileName(self, "Select Input Image", os.getenv("HOME"), filter)
        if fileName:
            self.lineEdit.setText(fileName[0])
            self.inputPath = self.lineEdit.text()

    def save_file(self):
        filter = "Images(*.jpg *.jpeg *.png)"
        fileName = QFileDialog.getSaveFileName(self, "Select Save Path", os.getenv("HOME"), filter)
        if fileName:
            self.lineEdit_2.setText(fileName[0])
            self.outputPath = self.lineEdit_2.text()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Window()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
