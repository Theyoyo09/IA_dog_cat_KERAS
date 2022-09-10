# set the matplotlib backend so figures can be saved in the background
import matplotlib

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import os

matplotlib.use("Agg")
