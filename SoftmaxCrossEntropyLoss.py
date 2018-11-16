#  This python script illustrates the different
#  loss functions for regression and classification.

import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a session 
sess = tf.Session()

# Softmax entropy loss,  L = -actual * (log(softmax(pred))) - (1-actual)(log(1-softmax(pred)))
rawscaled_logits = tf.constant([[1., -3., 10., 0.33, 8.2]])
##true_dist = tf.constant([[0.1, 0.02, 0.88]])
true_dist = tf.constant([[0.1, 0.02, 0.88, 0.4, 0.5]])
softmax_cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=rawscaled_logits,
                                                              labels=true_dist)
print(sess.run(softmax_cross_entropy_loss))

# Sparse entropy loss
# Use when classes and targets have to be mutually exclusive
# L = sum( -actual * log(predict) )
rawscaled_logits = tf.constant([[1., -3., 10.]])
sparse_true_dist = tf.constant([2])
sparse_cross_entropy =  tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rawscaled_logits,
                                                                  labels=sparse_true_dist)
print(sess.run(sparse_cross_entropy))