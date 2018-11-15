# Layering Nested Operations

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

# Create the data and variables
my_array = np.array([[11., 13., 15., 17., 19.],
                   [-12., 10., 2., 5., 7.],
                   [-3., 2., 1., 6., 5.]])

x_nums = np.array([my_array+5, my_array*10, my_array/3, my_array])
print(sess.run(tf.shape(x_nums)))  ###x_nums has a shape [4, 3, 5]

x_pldata = tf.placeholder(tf.float32, shape=(3, 5))
print(sess.run(tf.shape(x_pldata)))  ###x_pldata has a shape [3, 5]

# Constants for matrix multiplication:
m10 = tf.constant([[11.], [10.], [1.], [-2.], [2.]])
print(sess.run(tf.shape(m10)))  ###m10 has a shape [5 1]

m11 = tf.constant([[6.]])
print(sess.run(tf.shape(m11)))  ###m11 has a shape [1 1]

m13 = tf.constant([[8.]])
print(sess.run(tf.shape(m13)))  ###m13 has a shape [1 1]

# Create our multiple operations
prod1 = tf.matmul(x_pldata, m10)
prod2 = tf.matmul(prod1, m11)
add1 = tf.add(prod2, m13)

# Now feed data through placeholder and print results
for x_num in x_nums:
    print(sess.run(add1, feed_dict={x_pldata: x_num}))


