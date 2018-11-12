# How to create Tensors
# This function introduces various ways to create
# tensors in TensorFlow

import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()
tensor1 = tf.zeros([2,5])
print(sess.run(tensor1))

# Declare a variable.
# A variable needs to be inialized
var1 = tf.Variable(tf.ones([2,5]))
sess.run(var1.initializer)
print(sess.run(var1))

# Different kinds of variables
row_dim = 2
col_dim = 3 

# Zero initialized variable
zero_var = tf.Variable(tf.zeros([row_dim, col_dim]))

# One initialized variable
ones_var = tf.Variable(tf.ones([row_dim, col_dim]))

# shaped like other variable
sess.run(zero_var.initializer)
sess.run(ones_var.initializer)
zero_similar = tf.Variable(tf.zeros_like(zero_var))
ones_similar = tf.Variable(tf.ones_like(ones_var))

sess.run(ones_similar.initializer)
sess.run(zero_similar.initializer)
print(sess.run(ones_similar))
print(sess.run(zero_similar))

# Fill a tensor constant with a constant 3.46
A_filled_var = tf.Variable(tf.fill([row_dim, col_dim], 3.46))
sess.run(A_filled_var.initializer)
print(sess.run(A_filled_var))

# Create a variable from a constant [2, 3, 6, 15, 11, 0, 4, 7]
A_const_var = tf.Variable(tf.constant([2, 3, 6, 15, 11, 0, 4, 7]))
sess.run(A_const_var.initializer)
print(sess.run(A_const_var))

# This can also be used to fill an array:
A_const_fill_var = tf.Variable(tf.constant(2.15, shape=[row_dim, col_dim]))
sess.run(A_const_fill_var.initializer)
print(sess.run(A_const_fill_var))

# Sequence generation
A_linear_sequence_var = tf.Variable(tf.linspace(start=0.0, stop=9.0, num=7)) 
sess.run(A_linear_sequence_var.initializer)
print(sess.run(A_linear_sequence_var))

A_sequence_var = tf.Variable(tf.range(start=-5, limit=22, delta=3))
sess.run(A_sequence_var.initializer)
print(sess.run(A_sequence_var))

# Random Numbers
# Create Random Normal Variable
A_rand_norm_var = tf.random_normal([row_dim, col_dim], mean=4.5, stddev=0.5)
print(sess.run(A_rand_norm_var))
