# Operations on a Computational Graph
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a graph
sess = tf.Session()

# Create data to feed in
x_nums1 = np.array([5., 7., 9., 11., 13., 15, 17])
print(x_nums1)
x_nums2 = range(7)
x_nums=x_nums1+x_nums2

x_pldata = tf.placeholder(tf.float32)
my_const = tf.constant(5.)

# Multiplication
my_multiply = tf.multiply(x_pldata, my_const)
for x_num in x_nums:
    print(sess.run(my_multiply, feed_dict={x_pldata: x_num}))

merged = tf.summary.merge_all()
if not os.path.exists('c:/tmp/tensorboardevent_logs/'):
    os.makedirs('c:/tmp/tensorboardevent_logs/')

my_writer = tf.summary.FileWriter('c:/tmp/tensorboardevent_logs/', sess.graph)
