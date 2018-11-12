# Activation Functions
# This function introduces activation
# functions in TensorFlow

# Load Libraries For Implementing Activation Functions
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

# Calculate various activation functions.

# Hyper Tangent activation
print(sess.run(tf.nn.tanh([-1., -0.5, 0., 1., 1.5])))

# Softsign activation
print(sess.run(tf.nn.softsign([-1., 0., 1., -1., -0.5, 0., 1., 1.5])))

# Sigmoid activation
print(sess.run(tf.nn.sigmoid([-1.5, 0., 1.55, -1.3, -0.5, 0., 1., 1.5])))

# Calculate popular ReLU activation
print(sess.run(tf.nn.relu([-2., 2., 15., -1.6, -2.5, 0.7, 1.4, 1.85])))

# Calculate ReLU-6 activation
print(sess.run(tf.nn.relu6([-2., 2.-1.2, 7.+2.2, -1./3.4, 3.3*2, -0.5, -0.8, 0.3, 1., 1.5])))

# Softplus activation
print(sess.run(tf.nn.softplus([-1.*3.3, 10./11, 1.+8.3])))

# Exponential linear activation
print(sess.run(tf.nn.elu([-1., 0., 1.-1., tf.log(4.5), tf.cos(0.4), tf.sin(1.2), tf.cosh(1.25)])))

