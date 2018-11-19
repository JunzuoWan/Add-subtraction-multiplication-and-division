# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

print(tf.add(17, 2))   
print(tf.add([11, 21, 3, 4], [23, 41, 8, 6])) ###[11+23, 21+41, 3+8, 4+6]=[34, 62, 11, 10]
print(tf.square(8)) ##8*8=64
print(tf.reduce_sum([1, 2, 3, 4, 5])) ###it is summation of 1+2+3+4+5
print(tf.encode_base64("The first program is to print hello world"))

# Operator overloading is also supported
print(tf.square(4)-tf.square(3)+tf.abs(-5))

x = tf.matmul([[1, 2, 5]], [[2, 5, 3], [2, 4, 7], [3, 6, 5]]) ### 1x3 matrix multiple 3x3 matrix is 1x3 matrix
y = tf.matmul([[2, 4, 5], [1, 2, 5]], [[12, 15, 13], [20, 4, 17], [13, 3, 8]]) ### 2x3 matrix multiple 3x3 matrix is 2x3 matrix
print(x.shape)
print(x.dtype)
print(y.shape)
print(y.dtype)

ndarray = np.ones([4, 3])

print("A TensorFlow operation will convert numpy array to a tensor.")
Atensor = tf.multiply(ndarray, 14)
Atensor1 = tf.add(ndarray, 4)
Atensor2 = tf.subtract(ndarray, 3)
Atensor3 = tf.divide(ndarray, 5)
print(Atensor)
print(Atensor1)
print(Atensor2)
print(Atensor3)


print("A NumPy operation converts a tensor to a numpy array.")
print(np.add(Atensor, 1))
print(np.subtract(Atensor, 2))
print(np.divide(Atensor, 4))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(Atensor.numpy())
