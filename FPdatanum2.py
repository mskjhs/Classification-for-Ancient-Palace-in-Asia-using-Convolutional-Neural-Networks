from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.layers import BatchNormalization, Activation
import sklearn.metrics
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16

# set dir
train_dir = './100/train'
test_dir = './100/test'
validation_dir = './100/validation'

# set image generators
train_datagen = ImageDataGenerator(rescale=1./255,
                    rotation_range=20, shear_range=0.1,
                    width_shift_range=0.1, height_shift_range=0.1,
                    zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        batch_size=10,
        class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(256, 256),
        batch_size=10,
        class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(256, 256),
        batch_size=10,
        class_mode='categorical')

# model definition
input_shape = [256, 256, 3]
def build_model():
    model=models.Sequential()
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=input_shape)
    conv_base.trainable = False
    model.add(conv_base)
    # MLP
    model.add(layers.Flatten())
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(512))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(layers.Dropout(0.25))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(3, activation='sigmoid'))
    # compile
    model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# main loop
import time
starttime = time.time()
num_epochs = 50
model = build_model()
model.summary()
conv_base = model.layers[0]
conv_base.summary()
history = model.fit_generator(train_generator,
                    epochs=num_epochs, steps_per_epoch=10,
                    validation_data=validation_generator, validation_steps=2)

# saving the model
model.save('FPdata_VGG16_100_2.h5')

# evaluation
train_loss, train_acc = model.evaluate_generator(train_generator)
test_loss, test_acc = model.evaluate(test_generator)
print('train_acc:', train_acc)
print('test_acc:', test_acc)
print('train_loss:', train_loss)
print('test_loss:', test_loss)
print("elapsed time (in sec): ", time.time()-starttime)


# visualization
def plot_acc(h, title="accuracy"):
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
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
plt.savefig('FPdata_VGG16_100_2.loss.png')
plt.clf()
plot_acc(history)
plt.savefig('FPdata_VGG16_100_2.accuracy.png')