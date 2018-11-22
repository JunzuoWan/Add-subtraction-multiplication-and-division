# This python function shows how to implement back (retro) propagation
# in a regression model.

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph and start a session
sess = tf.Session()

# Here in this example we discuss regression:
# The sample data are created as follows:

# x-sample: 300 random samples from a normal ~ N(2, 0.1)
# target: 300 values of the value 18.

# We will fit the model:
# x-sample * W = target. In theroy, W = 9.

# Now we create 300 data for x-values.
x_input = np.random.normal(2, 0.2, 300)

###Now we create 300 target data for y-values.
y_out = np.repeat(18., 300)
##print(y_out) ###y_out are 300 numbers of 18.

###Introduce 2 placeholders for the training data
x_sample = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)

# Create variable W. In this model we only have one parameter W.
W = tf.Variable(tf.random_normal(shape=[1]))

# Add a multiplication operation to graph
y_output = tf.multiply(x_sample, W)

# Add L2 loss operation to graph
loss = tf.square(y_output - y_target)

# Create 2 Optimizers
theOptimizer1 = tf.train.GradientDescentOptimizer(0.0002)
theOptimizer2=tf.train.AdamOptimizer(0.002)
train_step1 = theOptimizer1.minimize(loss)
train_step2 = theOptimizer2.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Now we run the for-Loop for 3500 times.
for i in range(3500): ##repeat this for-loop 3500 times, 0, 1, 2,....3499.
    rand_index = np.random.choice(300)  
    ###In every for-loop, create a random number between 0 and 299
   # print(rand_index)
    rand_x = [x_input[rand_index]]
    rand_y = [y_out[rand_index]]
    sess.run(train_step1, feed_dict={x_sample: rand_x, y_target: rand_y})
    if (i+1)%5==0:
        print('Step #' + str(i+1) + 'GD W = ' + str(sess.run(W)))
        print('GDOpt Loss = ' + str(sess.run(loss, feed_dict={x_sample: rand_x, y_target: rand_y})))
    sess.run(train_step2, feed_dict={x_sample: rand_x, y_target: rand_y})
    if (i+1)%5==0:
        print('Step #' + str(i+1) + 'Adam W = ' + str(sess.run(W)))
        print('AdamOpt Loss = ' + str(sess.run(loss, feed_dict={x_sample: rand_x, y_target: rand_y})))


