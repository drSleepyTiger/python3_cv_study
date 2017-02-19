import tensorflow as tf
import numpy as np

dataset = np.loadtxt('train.txt', unpack=True)

x_data = dataset[0:-1]
y_data = dataset[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1,len(x_data)],-1.0, 1.0))

# hypothesis
h = tf.matmul(W, X)
hypothesis = tf.div(1.0, 1.0 + tf. exp(-h)) # sigmoid function

# cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1 - hypothesis))

# Minimize
a = tf.Variable(0.1)            #learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Initializing
init = tf.initialize_all_variables()

# Session run
with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(train, feed_dict={X:x_data, Y: y_data})
        if step % 20 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y: y_data}), sess.run(W))

    # Accuracy check
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)          # floor with + 0.5 : 반올림
                                                                        # hypothesis를 통한 추정과 Y 값이 같은지 비교
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
    print (sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data}))
    print ("Accuracy:", accuracy.eval(feed_dict={X:x_data, Y:y_data}))

