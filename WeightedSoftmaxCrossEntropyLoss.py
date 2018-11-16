#  This python script discusses the weighted softmax cross-entropy loss function

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

## Start a session 
sess = tf.Session()

###### Categorical Predictions ######
x_nums = tf.linspace(-3., 5., 600)
target = tf.constant(1.)
targets = tf.fill([600,], 1.)

# Weighted (softmax) cross entropy loss
# L = -actual * (log(pred)) * weights - (1-actual)(log(1-pred))
# or L = (1 - pred) * actual + (1 + (weights - 1) * pred) * log(1 + exp(-actual))
weight = tf.constant(0.4)
crossentropy_weighted_y = tf.nn.weighted_cross_entropy_with_logits(logits=x_nums,
                                                                    targets=targets,
                                                                    pos_weight=weight)
crossentropy_weighted_yo = sess.run(crossentropy_weighted_y)

# Plot the weighted cross entropy loss 
x_array = sess.run(x_nums)
plt.plot(x_array, crossentropy_weighted_yo, 'r:', label='Weighted Cross Entropy Loss (x0.4)')
plt.ylim(-0.1, 1.55)
plt.xlim(-3.0, 5.5)
plt.grid()
plt.legend(loc='upper right', prop={'size': 12})
plt.show()

