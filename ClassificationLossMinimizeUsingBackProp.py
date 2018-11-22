# Back (retro) Propagation
# This python function shows how to implement back (retro) propagation
# in a classification model.

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph and start a session
sess = tf.Session()

# This is a classification example.
# There are 200 values of the corresponding output index
# We will fit the binary classification model:
# If sigmoid(x+b) < 0.5 -> 0 else 1
# Theoretically, the constant bias b should be -(mean1 + mean2)/2

ops.reset_default_graph()

# Create graph
sess = tf.Session()

# We first create 100 sample data which are random values from a normal = N(-1, 0.2)
#np.concatenate((np.random.normal(-1, 0.2, 100), np.random.normal(4, 0.3, 110)))

x_sample1=np.random.normal(-1, 0.2, 100)
x_sample2=np.random.normal(4, 0.3, 110)
x_num = np.concatenate((x_sample1, x_sample2))


## we now create 100 values of 0
y_target1=np.repeat(0., 100)

##next we create 110 values of 1
y_target2=np.repeat(1., 110)

#y_vals1 = np.concatenate((np.repeat(0., 100), np.repeat(1., 110)))
y_num=np.concatenate((y_target1, y_target2))

print(x_num)
print(y_num)

##Now we create 2 placeholders
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)

# We now create a variable called bias (one model parameter = b)
b = tf.Variable(tf.random_normal(mean=5, shape=[1]))

#Next we create the activation operation using sigmoid function: sigmoid(x + b)
# The sigmoid() is the non-linear, activation part of the final loss function
y_out = tf.add(x_data, b)

# Now we have to add another dimension to each (batch size of 1)
y_out_expanded = tf.expand_dims(y_out, 0)
y_target_expanded = tf.expand_dims(y_target, 0)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Now calculate the classification loss which typically uses the cross entropy loss 
crossentropyloss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_out_expanded, labels=y_target_expanded)

# Next we define the Optimizer
theOptimizer = tf.train.GradientDescentOptimizer(0.04)
train_step = theOptimizer.minimize(crossentropyloss)

# USe for-loop to start the training...the following for-loop will run 1800 times.
for i in range(1800):
    rand_index = np.random.choice(210)  ##0 to 209 
    rand_x = [x_num[rand_index]]
    rand_y = [y_num[rand_index]]
    
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%100==0:
        print('Step#' + str(i+1) + ' b=' + str(sess.run(b)))
        print('Loss=' + str(sess.run(crossentropyloss, feed_dict={x_data: rand_x, y_target: rand_y})))

# Now it is time to evaluate predictions
predictions = [] ###empty list
for i in range(len(x_num)):  ##len() function returns total data number for x_num.
    x_val = [x_num[i]]
    prediction = sess.run(tf.round(tf.sigmoid(y_out)), feed_dict={x_data: x_val})
    predictions.append(prediction[0])
    
accuracy = sum(x==y for x,y in zip(predictions, y_num))/210.
print('Final Achieved Accuracy = ' + str(np.round(accuracy, 2)))