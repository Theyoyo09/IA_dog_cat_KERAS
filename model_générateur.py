# set the matplotlib backend so figures can be saved in the background
import matplotlib
from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

matplotlib.use("Agg")

image_entrainement = "Training/"
image_validation = "validation/"

image_chien = "Training/dogs"
image_chat = "Training/cats"

test = "test1/"

validation_chien = "validation/dogs"
validation_chat = "validation/cats"

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

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
optimizer=optimizers.RMSprop(learning_rate=1e-4),
metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

history = model.fit(
 train_generator,
 steps_per_epoch=150,
 epochs=150, callbacks=[es, mc],
 validation_data=validation_generator,
 validation_steps=50)

model.save('Mon_model_chien_chat.h5')
