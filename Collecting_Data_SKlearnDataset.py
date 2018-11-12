# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:09:03 2018
The program is mainly based on https://github.com/nfmcclure/tensorflow_cookbook
"""

#import matplotlib.pyplot as plt
#import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()


# Iris Data
from sklearn import datasets

iris = datasets.load_iris()
print(len(iris.data))
print(len(iris.target))
print(iris.data[0])
print(set(iris.target))
