#  This python script illustrates the hinge loss function 

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph
sess = tf.Session()

###### Numerical Predictions ######
x_predic = tf.linspace(-2.5, 4.8, 700)
x_true = tf.constant(0.8)

# Hinge loss is used for predicting binary (-1, 1) classes
# L = max(0, 1-(prediction*actual))
hinge_ys = tf.maximum(0., 1.0-tf.multiply(x_predic, x_true))
hinge_yo = sess.run(hinge_ys)

# Plot the Hinge Loss
x_val = sess.run(x_predic)
plt.plot(x_val, hinge_yo, 'b-', label='Hinge Loss') ##b is blue color
plt.ylim(-0.5, 3.5)
plt.xlim(-2.0, 5.0)
plt.grid()
plt.legend(loc='upper right', prop={'size': 12})
plt.show()
