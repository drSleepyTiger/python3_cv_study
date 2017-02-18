# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np

# Unpack 을 사용하면 Transpose 된 배열이 return 됨
dataset = np.loadtxt('train.txt', delimiter=',', unpack=True, skiprows=1, dtype=np.float32)
x_data = dataset[:-1, :]
y_data = dataset[-1, :]


# x_data = [[1.,1.,1.,1.,1.],
#           [1.,0.,3.,0.,5.],
#           [0.,2.,0.,4.,0.]]
# y_data = [1.,2.,3.,4.,5.]

W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))
# b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# Our hypothesis
# hypothesis = tf.matmul(W, x_data) + b
hypothesis = tf.matmul(W, x_data)

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimizing training
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Before starting, initialing the variables.
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        # print (step, sess.run(cost),sess.run(W), sess.run(b))
        print(step, sess.run(cost), sess.run(W))
