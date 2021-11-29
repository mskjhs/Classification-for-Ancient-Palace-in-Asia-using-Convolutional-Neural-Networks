from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
    # QAction, QApplication,QMainWindow ,QWidget, QPushButton, QMessageBox, QLabel, QLineEdit, QTextEdit, QGridLayout, QListWidget, QTableWidget,QFileDialog
from PyQt5.QtGui import *



class Window(QWidget):

    def __init__(self):
        super().__init__()
        self.title = "고궁판별기"
        self.top = 200
        self.left = 500
        self.width = 400
        self.height = 300
        self.InitWindow()

    def InitWindow(self):
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        self.text_edit = QTextEdit(self)
        self.text_edit.setGeometry(120, 200, 200, 50)
        self.setGeometry(self.left, self.top, self.width, self.height)
        vbox = QVBoxLayout()
        self.btn1 = QPushButton("Open Image")
        self.btn1.clicked.connect(self.getImage)
        vbox.addWidget(self.btn1)
        self.label = QLabel("result")
        vbox.addWidget(self.label)
        vbox.addWidget(self.text_edit)
        self.setLayout(vbox)
        self.show()

    def getImage(self):
        fname,_ = QFileDialog.getOpenFileName(self)
        imagePath = fname
        pixmap = QPixmap(imagePath).scaled(350,300)
        self.label.setPixmap(QPixmap(pixmap))
        self.resize(pixmap.width(), pixmap.height())
        if fname:
         data = fname
         print(data)
         #data500_VGG16_flatten_Dropout_model
         # model = load_model('projects_pretrained_change_setps_and_val_steps4(epoch50_and_data(each500)flatten_Dropout_)before_fine_tuning.h5')
         model = load_model('projects_pretrained_change_setps_and_val_steps4(256size)_add_class3(epoch50_and_data(each100)flatten_adam_softmax)before_fine_tuning.h5')
         # model =load_model('projects_pretrained_change_setps_and_val_steps4(128size)_add_class3(epoch50_and_data(each100)flatten_Dropout_adam_softmax)before_fine_tuning.h5')
         #data800_VGG16_flatten_Dropout_no_argumnet_model
         # model = load_model('projects_pretrained(Dropout_epoch50_and_data(each800)no_data_argu)before_fine_tuning.h5')projects_pretrained_change_setps_and_val_steps4(128size)_add_class3(epoch50_and_data(each100)flatten_Dropout_adam_softmax)before_fine_tuning

         #data100_VGG16_flatten
         # model =load_model('projects_pretrained_change_setps_and_val_steps4(epoch50_and_data(each100)flatten)before_fine_tuning.h5')
         img_path = data
         img = cv2.imread(data)
         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         img = image.load_img(img_path, target_size=(256, 256))
         img_tensor = image.img_to_array(img)
         img_tensor = np.expand_dims(img_tensor, axis=0)
         img_tensor = preprocess_input(img_tensor)
         pre = model.predict(img_tensor)
         merge = ('china:',round(pre[0][0]*100,2),'%','japan:',round(pre[0][1]*100,2),'%','korea:',round(pre[0][2]*100,2),'%')
         self.text_edit.setText(str(merge))


App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())