from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, optimizers, layers
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, GlobalAveragePooling2D, Input,Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.models import load_model,Model
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
import sklearn.metrics
import numpy as np
# set image generators
train_dir='/home/mskjhs/PycharmProjects/untitled2/projects_real/castle/train'
test_dir='/home/mskjhs/PycharmProjects/untitled2/projects_real/castle/test'
validation_dir='/home/mskjhs/PycharmProjects/untitled2/projects_real/castle/validation'

train_datagen = ImageDataGenerator(rescale=1./255,
                    rotation_range=20, shear_range=0.1,
                    width_shift_range=0.1, height_shift_range=0.1,
                    zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        batch_size=20,
        class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(256, 256),
        batch_size=20,
        class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(256, 256),
        batch_size=20,
        class_mode='categorical')


input_shape = [256, 256, 3] # as a shape of image
num_classes = 3
def build_model():
    model = models.Sequential()
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=input_shape)
    conv_base.trainable=False
    model.add(conv_base)
    model.add(layers.Flatten())
    # model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.25))
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(256, activation ='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # compile
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

import time
starttime=time.time()
num_epochs = 50
model = build_model()
model.summary()
conv_base = model.layers[0]
conv_base.summary()
history = model.fit_generator(train_generator,
                    epochs=num_epochs, steps_per_epoch=9,
                    validation_data=validation_generator, validation_steps=3)

# saving the model
model.save('projects_pretrained_change_setps_and_val_steps4(256size)_add_class3(epoch50_and_data(each100)flatten_Dropout_Batchno_adam_softmax)before_fine_tuning.h5')


train_loss, train_acc = model.evaluate_generator(train_generator)
test_loss, test_acc = model.evaluate(test_generator)
print('train_acc:', train_acc)
print('train_loss:',train_loss)
print('test_acc:', test_acc)
print('test_loos:',test_loss)
print("elapsed time (in sec): ", time.time()-starttime)


def plot_acc(h, title="accuracy"):
    plt.plot(h.history['acc'])
    plt.plot(h.history ['val_acc'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

def plot_loss(h, title="loss"):
    plt.plot(h.history ['loss'])
    plt.plot(h.history ['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

plot_loss(history)
plt.savefig('projects_pretrained_change_setps_and_val_steps4(256size)_add_class3(epoch50_and_data(each100)flatten_Dropout_Batchno_adam_softmax)before_fine_tuning.loss.png')
plt.clf()
plot_acc(history)
plt.savefig('projects_pretrained_change_setps_and_val_steps4(256size)_add_class3(epoch50_and_data(each100)flatten_Dropout_Batchno_adam_softmax)before_fine_tuning.accuracy.png')


