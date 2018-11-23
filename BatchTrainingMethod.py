# In this program we study the batch stochastic training process. We will use
# a regression model that uses one variable for the model.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
ops.reset_default_graph()

# We will implement a regression example in batch training.
# Now let us begin. In the first step, we create session graph
sess = tf.Session()

# In the second step, we declare batch size
batch_size = 25

# In the third step, we create 400 data for x-values and 400 data for y-values.
### these data will be used for the training process.

x_nums=np.random.normal(1, 0.15, 400)
y_nums=np.random.normal(15, 0.03, 400)

###the above x_nums and y_nums will be used for the training of the model.

for i in range(400):
    if (i+1)%20==0:
        print(x_nums)  ##print out some of these 400 randomly generated numbers.
        print(y_nums)  ##print out some of these 400 numbers of roughly 15.

###Now we create 2 placeholders for the feed_dict during the training of the model. 
x_pldata = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_pltarget = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create the variable W for the model. This model only has one parameter W.
W = tf.Variable(tf.random_normal(shape=[1,1]))

# Add operation to the session graph
y_predict = tf.matmul(x_pldata, W)

###################
# Add L2 loss operation to graph
L2loss = tf.reduce_mean(tf.square(y_predict-y_pltarget))

# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(L2loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

batchLoss = []
# Now we use for-loop to run the training..
for i in range(400):
    rnd_idx = np.random.choice(400, size=batch_size)
    rnd_x = np.transpose([x_nums[rnd_idx]])
    rnd_y = np.transpose([y_nums[rnd_idx]])
    sess.run(train_step, feed_dict={x_pldata: rnd_x, y_pltarget: rnd_y})
    if (i+1)%10==0:
        print('For Loop #' + str(i+1) + ' W= ' + str(sess.run(W)))
        training_loss = sess.run(L2loss, feed_dict={x_pldata: rnd_x, y_pltarget: rnd_y})
        print('Training Loss = ' + str(training_loss))
        batchLoss.append(training_loss)
        
plt.plot(range(0, 400, 10), batchLoss, 'g--', label='Loss from Batch Process, size=25')
plt.legend(loc='upper center', prop={'size': 13})
plt.show()
####the end

