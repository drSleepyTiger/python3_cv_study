import tensorflow as tf
import numpy as np

dataset = np.loadtxt('train.txt')

x_data = dataset[:,0:-1]
y_data = dataset[:,-1:]

# print (x_data, x_data.shape)
# print (y_data, y_data.shape)

X = tf.placeholder(tf.float32, name = 'X-input')
Y = tf.placeholder(tf.float32, name = 'Y-input')

# Deep network configuration.: Use more layers.
W1 = tf.Variable(tf.random_uniform([2,5],-1.0, 1.0), name = 'Weight1')
W2 = tf.Variable(tf.random_uniform([5,4],-1.0, 1.0), name = 'Weight2')
W3 = tf.Variable(tf.random_uniform([4,1],-1.0, 1.0), name = 'Weight3')

b1 = tf.Variable(tf.zeros([5]), name = 'Bias1')
b2 = tf.Variable(tf.zeros([4]), name = 'Bias2')
b3 = tf.Variable(tf.zeros([1]), name = 'Bias3')

# Add histogram
w1_hist = tf.summary.histogram('weights1', W1)
w2_hist = tf.summary.histogram('weights2', W2)
w3_hist = tf.summary.histogram('weights3', W2)

b1_hist = tf.summary.histogram('biases1', b1)
b2_hist = tf.summary.histogram('biases2', b2)
b3_hist = tf.summary.histogram('biases3', b3)

y_hist = tf.summary.histogram('y', Y)


# hypothesis
with tf.name_scope('layer2'):
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)

with tf.name_scope('layer3'):
    L3 = tf.sigmoid(tf.matmul(L2,W2) + b2)

with tf.name_scope('layer3'):
    hypothesis = tf.sigmoid(tf.matmul(L3,W3) + b3) # sigmoid function

# cost function
with tf.name_scope('cost'):
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1 - hypothesis))
    cost_sum = tf.summary.scalar("cost", cost)

# Minimize
with tf.name_scope('train'):
    a = tf.Variable(0.01)            #learning rate alpha
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)

# Initializing
init = tf.global_variables_initializer()

# Session run
with tf.Session() as sess:
    # tensorboard --logdir=./log/xor_logs                띄어쓰기 주의할 것
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./log/xor_logs", sess.graph)

    sess.run(init)

    for step in range(200001):
        sess.run(train, feed_dict={X:x_data, Y: y_data})

        if step % 2000 == 0:
            summary, _ = sess.run([merged,train], feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary, step)
            print('Step:',step, 'cost:', sess.run(cost, feed_dict={X:x_data, Y: y_data}))
            # print('\tW1, b1:',sess.run(W1),sess.run(b1))
            # print('\tW2, b2:',sess.run(W2),sess.run(b2))

    # Accuracy check
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)          # floor with + 0.5 : 반올림
                                                                        # hypothesis를 통한 추정과 Y 값이 같은지 비교
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
    print (sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data}))
    print ("Accuracy:", accuracy.eval(feed_dict={X:x_data, Y:y_data}))
