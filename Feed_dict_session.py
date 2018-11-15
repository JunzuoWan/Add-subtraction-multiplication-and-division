# Operations on a Computational Graph

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create a session 
sess = tf.Session()

# Create data to feed in
x_nums = np.array([1., 3., 5., 7., 9., 11., 13., 15., 17.])
x_pldata = tf.placeholder(tf.float32)
my_const = tf.constant(15.)

# Multiplication
my_multiply = tf.multiply(x_pldata, my_const)
for x_num in x_nums:
    print(sess.run(my_multiply, feed_dict={x_pldata: x_num}))

