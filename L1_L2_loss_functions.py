#  This python script illustrates the different
#  loss functions for regression and classification.

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a session 
sess = tf.Session()

###### Numerical Predictions for x-values######
x_nums = tf.linspace(0., 6., 1000) ##start, stop, number of points
x_true = tf.constant(3.)

# L2 loss, L = (x_prediction - x_true)^2
L2_nums = tf.square(x_nums-x_true)
L2_out = sess.run(L2_nums)

# L1 loss, L = abs(x_true-X_prediction)
L1_nums = tf.abs(x_true - x_nums)
L1_out = sess.run(L1_nums)

# Now plot the data:
x_values = sess.run(x_nums)
plt.plot(x_values, L2_out, 'k-.', label='L2 Loss')
plt.plot(x_values, L1_out, 'g:', label='L1 Loss')
plt.ylim(-0.1, 10.0)
plt.legend(loc='upper center', prop={'size': 12})
plt.grid()
plt.show()

