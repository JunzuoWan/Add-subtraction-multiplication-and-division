# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:20:54 2018
The program is mainly based on https://github.com/nfmcclure/tensorflow_cookbook
"""
# Data Gathering
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# CIFAR-10 Image Category Dataset
# The CIFAR-10 data ( https://www.cs.toronto.edu/~kriz/cifar.html ) contains 60,000 32x32 color images of 10 classes.
# It was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
# Alex Krizhevsky maintains the page referenced here.
# This is such a common dataset, that there are built in functions in TensorFlow to access this data.
# Running this command requires an internet connection and a few minutes to download all the images.

(X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.cifar10.load_data()

print(X_train.shape)
print(y_train.shape)
print(y_train[1,])  

# Plot the 1-th image 
from PIL import Image
img = Image.fromarray(X_train[1,:,:,:])
plt.imshow(img)
