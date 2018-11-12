# Activation Functions: This function introduces activation functions in TensorFlow

# First Load useful libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

# Define X range. Start, Stop and total number.
x_num = np.linspace(start=-15., stop=15., num=100)

# Calculate various activation functions.
y_softplus = sess.run(tf.nn.softplus(x_num))
y_softsign = sess.run(tf.nn.softsign(x_num))
y_elu = sess.run(tf.nn.elu(x_num))
y_sigmoid = sess.run(tf.nn.sigmoid(x_num))
y_tanh = sess.run(tf.nn.tanh(x_num))
y_relu = sess.run(tf.nn.relu(x_num))
y_relu6 = sess.run(tf.nn.relu6(x_num))

# Plot the above different functions
plt.plot(x_num, y_relu, 'b:', label='ReLU', linewidth=2.5)
plt.plot(x_num, y_elu, 'k-', label='ExpLU', linewidth=1.2)
plt.ylim([-1.5,17]) ##ymin, ymax.
plt.legend(loc='upper left')
plt.show()

plt.plot(x_num, y_softplus, 'r--', label='Softplus', linewidth=2.5)
plt.plot(x_num, y_relu6, 'g-.', label='ReLU6', linewidth=2.5)
plt.ylim([-1.5,17]) ##ymin, ymax.
plt.legend(loc='upper left')
plt.show()

plt.plot(x_num, y_sigmoid, 'r--', label='Sigmoid', linewidth=2.5)
plt.plot(x_num, y_tanh, 'b:', label='Tanh', linewidth=2.5)
plt.plot(x_num, y_softsign, 'g-.', label='Softsign', linewidth=2.5)
plt.ylim([-1.5,1.8])
plt.legend(loc='upper left')
plt.show()


