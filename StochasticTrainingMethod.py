# In this program we study the stochastic training process.
# We will use a regression model that uses one variable for the model.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
ops.reset_default_graph()

# We will implement a regression example in stochastic training.

# Now let us begin. In the first step, we create graph
sess = tf.Session()

# in the second step, we create 400 data for x-values and 400 data for y-values.
### these data will be used for the training process.
x_nums=np.random.normal(1, 0.15, 400)
y_nums=np.random.normal(15, 0.03, 400)
#y_nums = np.repeat(15., 400) ##this will create 400 data of the same value 15.0
###the above x_nums and y_nums will be used for the training of the model.

for i in range(400):
    if (i+1)%20==0:
        print(x_nums)  ##print out some of these 400 randomly generated numbers.
        print(y_nums)  ##print out some of these 400 numbers of 15.

###Now we create 2 placeholders for the feed_dict during the training of the model. 
x_pldata = tf.placeholder(shape=[1], dtype=tf.float32)
y_pltarget = tf.placeholder(shape=[1], dtype=tf.float32)

# Create the variable W for the model. This model only has one parameter W.
W = tf.Variable(tf.random_normal(shape=[1]))

# Add operation to the session graph
y_predict = tf.multiply(x_pldata, W)

# Add L2 loss operation to the session graph
L2loss = tf.square(y_predict-y_pltarget)

# Create the Optimizer that uses the stochastic training process.
theStochasOptimizer = tf.train.GradientDescentOptimizer(0.02)
train_step = theStochasOptimizer.minimize(L2loss)

# Initialize global variables
init = tf.global_variables_initializer()
sess.run(init)

stochastictrain = [] ##create empty list
# Start training process. Run the for-loop 400 times.
for i in range(400):
    rnd_idx = np.random.choice(400) ###create a number between 0 and 399.
    rnd_x = [x_nums[rnd_idx]]
    rnd_y = [y_nums[rnd_idx]]
    sess.run(train_step, feed_dict={x_pldata: rnd_x, y_pltarget: rnd_y})
    if (i+1)%2==0:
        print('For Loop #' + str(i+1) + ' W=' + str(sess.run(W)))
        stoch_train_loss = sess.run(L2loss, feed_dict={x_pldata: rnd_x, y_pltarget: rnd_y})
        print('L2 Loss = ' + str(stoch_train_loss))
        stochastictrain.append(stoch_train_loss)
        
print("***************************") ##this will print a saparation line between the stochastic training 
##method and the graph plot.
        
plt.plot(range(0, 400, 2), stochastictrain, 'g-', label='Loss from Stochastic training')
plt.legend(loc='upper center', prop={'size': 13})
plt.show()