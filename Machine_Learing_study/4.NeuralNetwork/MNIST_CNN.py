import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

def xavier_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


# Parameters
learning_rate = 0.001
training_epochs = 8000
batch_size = 100
display_step = 1

# tf Graph Input
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# Create model

W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
W3 = tf.Variable(tf.random_normal([3,3,64,128], stddev=0.01))
W4 = tf.Variable(tf.random_normal([128*4*4,625], stddev=0.01))
W5 = tf.Variable(tf.random_normal([625,10], stddev=0.01))

# Construct model
dropout_cnn_rate = tf.placeholder(tf.float32)
dropout_fcc_rate = tf.placeholder(tf.float32)
X_image = tf.reshape(X, [-1, 28, 28, 1], name = 'X-input-reshape')

l1a = tf.nn.relu(tf.nn.conv2d(X_image,W1,strides=[1,1,1,1], padding="SAME"))
print(l1a)
l1 = tf.nn.max_pool(l1a, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
print(l1)
l1 = tf.nn.dropout(l1,dropout_cnn_rate)

l2a = tf.nn.relu(tf.nn.conv2d(l1,W2,strides=[1,1,1,1], padding="SAME"))
l2 = tf.nn.max_pool(l2a, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
l2 = tf.nn.dropout(l2,dropout_cnn_rate)

l3a = tf.nn.relu(tf.nn.conv2d(l2,W3,strides=[1,1,1,1], padding="SAME"))
l3 = tf.nn.max_pool(l3a, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
l3 = tf.reshape(l3,[-1,W4.get_shape().as_list()[0]])
l3 = tf.nn.dropout(l3,dropout_cnn_rate)

l4 = tf.nn.relu(tf.matmul(l3,W4))
l4 = tf.nn.dropout(l4,dropout_fcc_rate)

hypothesis = tf.matmul(l4,W5)


# Minimize error using cross entropy
cost = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(_sentinel=None, labels=Y, logits=hypothesis, dim=-1)))    # cross-entropy with Solftmax loss
optimizer = tf.train.RMSPropOptimizer(learning_rate,0.9).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1)), tf.float32))


# Initializing the variable
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    #Train cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        print ('total_batch:',total_batch)
        #Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys, dropout_cnn_rate: 0.7, dropout_fcc_rate: 0.5})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys, dropout_cnn_rate: 0.7, dropout_fcc_rate: 0.5})/total_batch

        #Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", "%04d" % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))

    # Calcuate accuracy
    print ("Accuracy:", accuracy.eval({X:mnist.test.images, Y: mnist.test.labels, dropout_cnn_rate: 1.0, dropout_fcc_rate: 1.0}))


#
#
