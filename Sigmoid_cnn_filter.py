# Working with Multiple cnn Layers
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create a graph
sess = tf.Session()

# Create a small random 'image' of size 4x4
rndimg_shape = [1, 4, 4, 1]
x_num = np.random.uniform(size=rndimg_shape)

x_pldata = tf.placeholder(tf.float32, shape=rndimg_shape)

# Create a layer that takes a spatial moving window average
# Our window will be 2x2 with a stride of 2 for height and width
# The filter value will be 0.2 because we want the average of the 2x2 window
thefilter = tf.constant(0.2, shape=[2, 2, 1, 1])
thestrides = [1, 2, 2, 1]
the_mov_avg_layer= tf.nn.conv2d(x_pldata, thefilter, thestrides,
                            padding='SAME', name='Move_Avg_Window')

# Define a function for a cnnm layer which will be sigmoid(xW+b) where
# x is a 2x2 matrix and W and b are 2x2 matrices
def cnn_layer(inputMatrixX):
    sqeezedMatrixX = tf.squeeze(inputMatrixX)
    print(sess.run(tf.shape(sqeezedMatrixX)))
    W = tf.constant([[0.5, 0.2], [-0.4, -0.3]])
    print(sess.run(tf.shape(W)))
    b = tf.constant(0.3, shape=[2, 2])
    print(sess.run(tf.shape(b)))
    XW = tf.matmul(sqeezedMatrixX,W)
    print(sess.run(tf.shape(XW)))
    XWb = tf.add(XW, b) # x*W + b
    return(tf.sigmoid(XWb)) ###use sigmoid activation function.

# Add a cnn layer to the graph
with tf.name_scope('cnn_Layer') as scope:
    cnn_layer1 = cnn_layer(the_mov_avg_layer)

# The output is now an array of 2x2, but size (1,2,2,1)
print(sess.run(the_mov_avg_layer, feed_dict={x_pldata: x_num}))

# After the operation, size changed to 2x2 (squeezed out size 1 dims)
print(sess.run(cnn_layer1, feed_dict={x_pldata: x_num}))

