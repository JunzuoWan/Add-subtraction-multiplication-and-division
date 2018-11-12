# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:11:48 2018
The program is based on https://github.com/nfmcclure/tensorflow_cookbook
"""

from tensorflow.python.framework import ops
ops.reset_default_graph()

# Housing Price Data
from keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print(x_train.shape[0])
print(x_train.shape[1])
