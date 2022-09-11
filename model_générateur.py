# set the matplotlib backend so figures can be saved in the background
import matplotlib

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import os

matplotlib.use("Agg")

image_entrainement = "C:/Users/Yoan/PycharmProjects/IA_dog_cat/Training/"
image_validation = "C:/Users/Yoan/PycharmProjects/IA_dog_cat/validation/"

image_chien = "C:/Users/Yoan/PycharmProjects/IA_dog_cat/Training/dogs"
image_chat = "C:/Users/Yoan/PycharmProjects/IA_dog_cat/Training/cats"

test = "C:/Users/Yoan/PycharmProjects/IA_dog_cat/test1/"

validation_chien = "C:/Users/Yoan/PycharmProjects/IA_dog_cat/validation/dogs"
validation_chat = "C:/Users/Yoan/PycharmProjects/IA_dog_cat/validation/cats"

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(image_entrainement,
 batch_size=20,
 target_size=(150, 150),
 class_mode='binary')

validation_generator = datagen.flow_from_directory(image_validation,
 batch_size=20,
 target_size=(150, 150),
 class_mode='binary')

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())

model.add(layers.Dropout(0.3))

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
optimizer=optimizers.RMSprop(lr=1e-4),
metrics=['acc'])

history = model.fit_generator(
 train_generator,
 steps_per_epoch=100,
 epochs=50,
 validation_data=validation_generator,
 validation_steps=50)

model.save('Mon_model_chien_chat.h5')

test_generator = datagen.flow_from_directory(
 test,
 target_size=(150, 150),
 batch_size=20,
 class_mode='binary')

model.evaluate(test_generator)