# This function introduces various math operations in TensorFlow

# Declaring Operations
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Open graph session
sess = tf.Session()

# This is div() 
print(sess.run(tf.div(12, 7)))

# This is truediv()
print(sess.run(tf.truediv(12, 7)))

# This is floordiv()
print(sess.run(tf.floordiv(3.54, 4.62)))

# This Mod function
print(sess.run(tf.mod(22.0, 7.0)))

# This is Cross Product
print(sess.run(tf.cross([1., 0., 0.], [0., 1., 0.])))



