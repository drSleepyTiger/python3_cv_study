import tensorflow as tf
import numpy as np

dataset = np.loadtxt('train.txt',dtype="float32")

x_data = dataset[:,0:-3]
y_data = dataset[:,-3:]


# Variable declare
X = tf.placeholder(tf.float32,[None,3])         # x0(bias), x1, x2 : 3 inputs
Y = tf.placeholder(tf.float32,[None,3])         # A, B, C          : 3 classes

# construct model
W = tf.Variable(tf.zeros([3,3]), name='weight')
# W = tf.Variable(tf.random_uniform([3, 3], -1.0, 1.0), name='weight')

# Hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X, W))

# Cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))

## reduction_indices 사용법
# 'x' is [[1, 1, 1]
#         [1, 1, 1]]
# tf.reduce_sum(x) ==> 6
# tf.reduce_sum(x, 0) ==> [2, 2, 2]
# tf.reduce_sum(x, 1) ==> [3, 3]
# tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
# tf.reduce_sum(x, [0, 1]) ==> 6

# Minimize
alpha = tf.Variable(0.001)            #learning rate
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

# Initializing
init = tf.global_variables_initializer()

# Session run
with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(train, feed_dict={X:x_data, Y: y_data})
        if step % 20 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y: y_data}), sess.run(W))

    # # 가설을 통한 검증
    # print ("--------------------------------------------")
    a = sess.run(hypothesis, feed_dict={X:[[1, 11, 7]]})
    print (a, sess.run(tf.arg_max(a,1)))

    b = sess.run(hypothesis, feed_dict={X:[[1, 3, 4]]})
    print (b, sess.run(tf.arg_max(b,1)))

    c = sess.run(hypothesis, feed_dict={X:[[1, 1, 0]]})
    print (c, sess.run(tf.arg_max(c,1)))

    all = sess.run(hypothesis, feed_dict={X:[[1, 11, 7],[1,3,4],[1,1,0]]})
    print (all, sess.run(tf.arg_max(all,1)))