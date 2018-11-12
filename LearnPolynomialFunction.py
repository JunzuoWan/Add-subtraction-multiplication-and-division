# This function introduces various operations in TensorFlow

# Import useful libraries
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# start a graph session
sess = tf.Session()

##define a polynomial function
def A_polynomial_func1(x_para):
    # Calculate and Return 5x^2 - 7*x + 13
    return tf.subtract(5 * tf.square(x_para), 7*x_para) + 13
print(sess.run(A_polynomial_func1(2)))

print("*************************")

##define a polynomial function
def A_polynomial_func2(x_para):
    # Calculate and Return 5x^2 + 6*x - 11
    return tf.add(5 * tf.square(x_para), 6*x_para) - 11
print(sess.run(A_polynomial_func2(7)))

print("*************************")

# define a range operation
testnums = range(19)
# calculate TensorFlow function output
for xn in testnums:
    print(sess.run(A_polynomial_func1(xn)))
