# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 10:21:56 2018

@author: J.Wan
"""

#linear regression
##linear regression: find the linear fit to generated data

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def createDataset():
    x_batch =np.linspace(-1.5,1.5,51)
    y_batch =5*x_batch+0.25*np.random.randn(*x_batch.shape)
    return x_batch, y_batch

def linearRegressionModel():
    x=tf.placeholder(tf.float32, shape=(None,), name='x')
    y=tf.placeholder(tf.float32, shape=(None,), name='y')
    w=tf.Variable(np.random.normal(), name='W')
    y_predict=tf.multiply(w,x)
    loss=tf.reduce_mean(tf.square(y_predict-y))      
    return x, y, y_predict, loss

def runFunc():
    x_batch, y_batch=createDataset()  
    x, y, y_predi, calculLoss=linearRegressionModel()
    optimizer=tf.train.GradientDescentOptimizer(0.07).minimize(calculLoss) 
    init=tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)      
        feed_dict={x: x_batch, y: y_batch}
        for _ in range(250): 
            loss_value, _=session.run([calculLoss, optimizer], feed_dict)
            print('Loss is:', loss_value.mean())
        y_pred_batch=session.run(y_predi,{x: x_batch})
    
    plt.figure(1)
    plt.scatter(x_batch, y_batch)
    plt.plot(x_batch,y_pred_batch)
    plt.xlabel("x data")
    plt.ylabel("y value")
    plt.title("Linear Regression")
    plt.savefig('LinearRegressionPlot.png')
    
if __name__=='__main__':
    runFunc()    

