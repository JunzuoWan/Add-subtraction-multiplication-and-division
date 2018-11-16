
###  This python script discusses the cross-entropy loss function

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a session 
sess = tf.Session()

###### Numerical Predictions for x-values######
x_nums = tf.linspace(-1., 1.3, 500) #start, stop, number of points
x_true = tf.constant(0.8)

# Define the graph model of calculate cross entropy loss: Loss = -x_true * (log(x_predict)) - (1-x_true)(log(1-x_predict))
cross_entropy_y = - tf.multiply(x_true, tf.log(x_nums)) - tf.multiply((1. - x_true), tf.log(1. - x_nums))
##obtain the cross_entropy values
cross_entropy_yo = sess.run(cross_entropy_y)


# Plot the cross_entropy loss vs x_vals
x_vals = sess.run(x_nums)
plt.plot(x_vals, cross_entropy_yo, 'b--', label='Cross Entropy Loss')
plt.ylim(-0.5, 4)
plt.xlim(-0.1, 1.1)
plt.grid()
plt.legend(loc='upper center', prop={'size': 14})
plt.show()
