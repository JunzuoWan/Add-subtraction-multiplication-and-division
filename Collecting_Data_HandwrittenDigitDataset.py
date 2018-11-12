# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:15:54 2018
The program is based on https://github.com/nfmcclure/tensorflow_cookbook
"""
from tensorflow.python.framework import ops
ops.reset_default_graph()

# MNIST Handwriting Data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(len(mnist.train.images))
print(len(mnist.test.images))
print(len(mnist.validation.images))
print(mnist.train.labels[1,:])
print(mnist.train.labels[5,:])
print(mnist.train.labels[10,:])
