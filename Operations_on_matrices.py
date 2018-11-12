# This program describes matrices and their Operations
# This function introduces various ways to create
# matrices and how to use them in TensorFlow

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Define a session
sess = tf.Session()

# Define Identity Matrix
A_identity_matrix = tf.diag([1.0,1.0,1.0])
print(sess.run(A_identity_matrix))

# Create a 3x3 Random Norm Matrix
A_matrix = tf.truncated_normal([3,3])
print(sess.run(A_matrix))

# Create a 3x2 Constant Matrix, 7.0 is the Constant Value
A_const_matrix = tf.fill([3,2], 7.0)
print(sess.run(A_const_matrix))

# Calculate a Matrix Multiplication
print(sess.run(tf.matmul(A_matrix, A_const_matrix)))

# Create 4x4 Random Uniform Matrix
A_rand_matrix = tf.random_uniform([4,4])
print(sess.run(A_rand_matrix))  

# Create a Matrix From a np Array
ATensor_from_array = tf.convert_to_tensor(np.array([[12.7, 21.5, 13.2], [-3.3, -17.2, -11.3], [0.1, 5.1, -2.7]]))
print(sess.run(ATensor_from_array))

# Calculate Matrix Addition and Subtraction
print(sess.run(A_identity_matrix+A_matrix))
print(sess.run(A_identity_matrix-A_matrix))

# Calculate Matrix Transpose
print(sess.run(tf.transpose(A_const_matrix))) 

# Calculate Matrix Determinant
print(sess.run(tf.matrix_determinant(A_rand_matrix)))

# Calculate Another Matrix Multiplication
print(sess.run(tf.matmul(A_matrix, A_identity_matrix)))

# Calculate Matrix Inverse
print(sess.run(tf.matrix_inverse(A_rand_matrix)))
print(sess.run(tf.matrix_inverse(A_matrix)))

# Calculate Eigenvalues and Eigenvectors
print(sess.run(tf.self_adjoint_eig(A_rand_matrix)))
print(sess.run(tf.self_adjoint_eig(A_matrix)))

