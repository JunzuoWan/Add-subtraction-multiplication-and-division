# Activation Functions: This function introduces activation
# functions in TensorFlow

# Load libraries
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

# Define X range. Start, Stop and total number.
x_num = np.linspace(start=-15., stop=15., num=8)

# Calculate various activation functions.
print(sess.run(tf.nn.softplus(x_num)))
print(sess.run(tf.nn.softsign(x_num)))
print(sess.run(tf.nn.elu(x_num)))
print(sess.run(tf.nn.sigmoid(x_num)))
print(sess.run(tf.nn.tanh(x_num)))
print(sess.run(tf.nn.relu(x_num)))
print(sess.run(tf.nn.relu6(x_num)))

