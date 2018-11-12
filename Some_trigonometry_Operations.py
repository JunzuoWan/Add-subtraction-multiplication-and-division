# Operations
#----------------------------------
#
# This function introduces various operations
# in TensorFlow

# Declaring Operations
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Open graph session
sess = tf.Session()

# Trig functions
print(sess.run(tf.sin(2.*3.1416)))
print(sess.run(tf.cos(3.1416/8.)))
print(sess.run(tf.tan(3.1416/4.)))
print(sess.run(tf.tanh(3.1416/2.)))
print(sess.run(tf.cosh(3.1416/16.)))
print(sess.run(tf.sinh(3.1416/5.)))

print("OOOOOOOOOO")

# Custom operation
for x in range(9):
    print(sess.run(tf.tan(3.1416/(x+1))))

print("UUUUUUUUUU")

for x in range(6):
    print(sess.run(tf.sinh(3.1416/(x+1))))

