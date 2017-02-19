import tensorflow as tf
import numpy as np

dataset = np.loadtxt('train.txt')

x_data = dataset[:,0:-1]
y_data = dataset[:,-1:]

# print (x_data, x_data.shape)
# print (y_data, y_data.shape)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2,2],-1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([2,1],-1.0, 1.0))

b1 = tf.Variable(tf.zeros([2]), name = 'Bias1')
b2 = tf.Variable(tf.zeros([1]), name = 'Bias2')

# hypothesis
L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
hypothesis = tf.sigmoid(tf.matmul(L2,W2) + b2) # sigmoid function

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

    for step in range(200001):
        sess.run(train, feed_dict={X:x_data, Y: y_data})
        if step % 2000 == 0:
            print('Step:',step, 'cost:', sess.run(cost, feed_dict={X:x_data, Y: y_data}))
            print('\tW1, b1:',sess.run(W1),sess.run(b1))
            print('\tW2, b2:',sess.run(W2),sess.run(b2))

    # Accuracy check
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)          # floor with + 0.5 : 반올림
                                                                        # hypothesis를 통한 추정과 Y 값이 같은지 비교
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
    print (sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data}))
    print ("Accuracy:", accuracy.eval(feed_dict={X:x_data, Y:y_data}))
