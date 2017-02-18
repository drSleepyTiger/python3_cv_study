import tensorflow as tf
import numpy as np

dataset = np.loadtxt('train.txt', unpack=True, dtype = 'float32', delimiter = ',', skiprows=1)

x_data = dataset[0:-1]
y_data = dataset[-1]


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform((1,len(x_data)), -1.,1.))

# Our hypothesis
z = tf.matmul(W,X)
hypothesis = tf.div(1.0, 1.0 + tf.exp(-z))

# Cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1 - hypothesis))

# Minimize
a = tf.Variable(0.1)            #learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Initializing
init = tf.initialize_all_variables()

# Session run
sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X:x_data, Y: y_data}), sess.run(W))

# 가설을 통한 검증
print ("--------------------------------------------")
print (sess.run(hypothesis, feed_dict={X:[[1],[2],[2]]}) > 0.5 )
print (sess.run(hypothesis, feed_dict={X:[[1],[5],[5]]}) > 0.5 )

print (sess.run(hypothesis, feed_dict={X:[[1,1],[2,5],[2,5]]}) > 0.5 )