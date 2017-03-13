import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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
training_epochs = 15
batch_size = 100
display_step = 1

# tf Graph Input
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# Create model
W1 = tf.get_variable("W1", shape=[784,256], initializer=xavier_init(784,256))
W2 = tf.get_variable("W2", shape=[256,256], initializer=xavier_init(256,256))
W3 = tf.get_variable("W3", shape=[256,256], initializer=xavier_init(256,256))
W4 = tf.get_variable("W4", shape=[256,256], initializer=xavier_init(256,256))
W5 = tf.get_variable("W5", shape=[256,10], initializer=xavier_init(256,10))


B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([256]))
B4 = tf.Variable(tf.random_normal([256]))
B5 = tf.Variable(tf.random_normal([10]))

# Construct model
dropout_rate = tf.placeholder("float")
_L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
L1 = tf.nn.dropout(_L1, dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))
L2 = tf.nn.dropout(_L2, dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), B3))
L3 = tf.nn.dropout(_L3, dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), B4))
L4 = tf.nn.dropout(_L4, dropout_rate)

hypothesis = tf.add(tf.matmul(L4, W5), B5)


# Minimize error using cross entropy
cost = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(_sentinel=None, labels=Y, logits=hypothesis, dim=-1)))    # cross-entropy with Solftmax loss
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis,Y))    # cross-entropy with Solftmax loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initializing the variable
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    #Train cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        #Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys, dropout_rate: 0.7})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys, dropout_rate: 0.7})/total_batch

        #Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", "%04d" % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))

    # Calcuate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({X:mnist.test.images, Y: mnist.test.labels, dropout_rate: 1.}))




