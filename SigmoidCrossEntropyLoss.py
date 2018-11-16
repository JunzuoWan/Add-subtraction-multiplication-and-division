#  This python script illustrates the sigmoid cross entropy loss 

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a session 
sess = tf.Session()

x_nums = tf.linspace(-3.2, 5.5, 650)
target = tf.constant(1.)
targets = tf.fill([650,], 1.)

x_val = tf.expand_dims(x_nums, 1)
target_input = tf.expand_dims(targets, 1)
crossentropy_sigmoid_y = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_val,
                                                                  labels=target_input)
crossentropy_sigmoid_yo = sess.run(crossentropy_sigmoid_y)

# Plot the entropy loss
xarray = sess.run(x_nums)
plt.plot(xarray, crossentropy_sigmoid_yo, 'b-.', label='Cross Entropy Sigmoid Loss')
plt.ylim(-0.5, 3.5)
plt.xlim(-4., 6.0)
plt.grid()
plt.legend(loc='upper right', prop={'size': 13})
plt.show()

