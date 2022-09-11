import cv2
import keras
import numpy as np
from PIL.ImageOps import grayscale
from tensorflow.python.ops.image_ops_impl import rgb_to_grayscale

Categories = ["cats", "dogs"]

def prepare():
    Img_size = 150
    img_array = cv2.imread('test1/test1/179.jpg')
    new_array = cv2.resize(img_array, (Img_size, Img_size))
    return new_array.reshape(-1, Img_size, Img_size, 3)


model = keras.models.load_model("Mon_model_chien_chat.h5")

prediction = model.predict([prepare()])
print(prediction)
print( Categories[int(prediction[0][0])])
