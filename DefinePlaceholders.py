# Placeholders
# This function describes how to 
# use placeholders in TensorFlow

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Using Placeholders
sess = tf.Session()

##define x as placeholder. Shape is 3x3.
x = tf.placeholder(tf.float32, shape=(3, 3))
##define identity
y = tf.identity(x)

##create a random array 3x3. 
A_rand_array = np.random.rand(3, 3)
print(sess.run(y, feed_dict={x: A_rand_array}))
