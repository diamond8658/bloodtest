import numpy as np
import tensorflow as tf
import itertools
import pandas as pd
import sys
from builtins import range, input
import os
from glob import glob
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications import VGG16
import tensorflow as tf
from keras.layers import Dense, Flatten, Input, Lambda, SeparableConv2D, BatchNormalization, Dropout, MaxPooling2D, Conv2D, Activation
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from google.colab import drive
drive.mount('/content/drive')

# Set path to directory containing dataset
train_path = 'content/drive/MyDrive/BloodTest/dataset2-master/images/TRAIN'
test_path = 'content/drive/MyDrive/BloodTest/dataset2-master/images/TEST'
train_images = glob(train_path + "/*/*.jp*g")
test_images = glob(test_path + "/*/*.jp*g")
folders = glob(train_path + "/*")
k = len(folders)
print(f"Number of Training samples: {len(train_images)}")
print(f"Number of Validation samples: {len(test_images)}")

# Plot an image from the dataset
plt.imshow(image.img_to_array(image.load_img(train_images[0])).astype('uint8'))
plt.show()

# Set up
class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
num_class = len(class_names)
image_size = [200, 200]
epochs = 5
batch_size = 32
activation = "relu"

# Data Generator
data_gen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.1,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True,
                             preprocessing_function=preprocess_input)

test_gen = data_gen.flow_from_directory(test_path, target_size=image_size)
print(test_gen.class_indices)

labels = [None] * len(test_gen.class_indices)
for k, v in test_gen.class_indices.items():
      labels[v] = k
      
train_generator = data_gen.flow_from_directory(train_path,
                                              target_size=image_size,
                                              shuffle=True,
                                              batch_size=batch_size)

test_generator = data_gen.flow_from_directory(test_path,
                                             target_size=image_size,
                                             shuffle=True,
                                             batch_size=batch_size)

# Setting up ResNet model Courtesy of: https://www.kaggle.com/code/shilpagopal/resnet50-classifier
resnet50 = ResNet50(input_shape=image_size + [3], weights = 'imagenet', include_top = False)
for layer in resnet50.layers:
      layer.trainable = False

x = Flatten()(resnet50.output)
pred = Dense(k, activation = 'softmax')

model = Model(inputs = resnet50.inputs, outputs = pred)
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

spe = int(len(train_images)//batch_size)
val = int(len(test_images)//batch_size)

res = model.fit_generator(train_generator,
                          validation_data=test_generator,
                          epochs=epochs,
                          steps_per_epoch=spe,
                          validation_steps= val,
                          verbose=1)

# Setting up VGG-16 Courtesy of: https://www.kaggle.com/code/kbrans/vgg16-model-83-36-acc
image_size = (150, 150)
epochs = 30
vgg_16 = VGG16(include_top = False, weights = 'imagenet', input_tensor = None, input_shape = (150,150,3), pooling = None)

vgg_16.trainable = True
for layer in vgg_16.layers:
  layer.trainable = False

input = Input(shape = (150, 150, 3))
layer = vgg_16(inputs=input)
layer = Flatten()(layer)
layer = BatchNormalization()(layer)
layer = Dense(units=256, activation='relu')(layer)
layer = Dropout(0.7)(layer)
layer = BatchNormalization()(layer)
layer = Dense(units=128, activation='relu')(layer)
layer = Dropout(0.5)(layer)
layer = Dense(units=64, activation='relu')(layer)
layer = Dropout(0.3)(layer)
layer = Dense(units=4,activation='softmax')(layer)

model2 = Model(inputs = input, outputs = layer)
model2.compile(loss='sparse_categorical_crossentropy',
            optimizer= 'adam',
            metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(
    monitor = 'val_accuracy', 
    patience = 2, 
    verbose = 1, 
    factor = 0.3, 
    min_lr = 0.000001)

res2 = model2.fit_generator(train_generator,
                          validation_data=test_generator,
                          epochs=epochs,
                          steps_per_epoch=spe,
                          validation_steps= val,
                          verbose=1)

# Setting up MobileNetV2 Model Courtesy of https://www.kaggle.com/code/stpeteishii/blood-cell-mobilenetv2-model
image_size = (120, 120)
mobilenet = tf.keras.applications.MobileNetV2(input_shape = (120, 120, 3), include_top = False, weights = 'imagenet', pooling='avg')
mobilenet.trainable = False

inputs = mobilenet.input
x = Dense(128, activation = 'relu')(mobilenet.output)
outputs = Dense(4, activation='softmax')
model3 = Model(inputs = inputs, outputs = outputs)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
res3 = model2.fit_generator(train_generator,
                          validation_data=test_generator,
                          epochs=epochs,
                          steps_per_epoch=spe,
                          validation_steps= val,
                          verbose=1)

# Plotting graphs for analysis
def get_confusion_matrix(data_path, N):
    # we need to see the data in the same order
    # for both predictions and targets
    predictions = []
    targets = []
    i = 0
    for x, y in data_gen.flow_from_directory(data_path, target_size=image_size, shuffle=False, batch_size=batch_size * 2):
        i += 1
        if i % 1000 == 0:
            print(i)
        p = model.predict(x)
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)
        predictions = np.concatenate((predictions, p))
        targets = np.concatenate((targets, y))
        if len(targets) >= N:
            break

    cm = confusion_matrix(targets, predictions)
    return cm

cm = get_confusion_matrix(train_path, len(train_images))
validation_cm = get_confusion_matrix(test_path, len(test_images))

# Plotting accuracies for models
plt.plot(res.history['accuracy'], label='acc')
plt.plot(res.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

plt.plot(res2.history['accuracy'], label='acc')
plt.plot(res2.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

plt.plot(res3.history['accuracy'], label='acc')
plt.plot(res3.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

#Plotting Loss for models
plt.plot(res.history['loss'], label='loss')
plt.plot(res.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(res2.history['loss'], label='loss')
plt.plot(res2.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(res3.history['loss'], label='loss')
plt.plot(res3.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def y2indicator(Y):
    K = len(set(Y))
    N = len(Y)
    I = np.empty((N, K))
    I[np.arange(N), Y] = 1
    return I

plot_confusion_matrix(cm, labels, title='Train confusion matrix')
plot_confusion_matrix(validation_cm, labels, title='Validation confusion matrix')