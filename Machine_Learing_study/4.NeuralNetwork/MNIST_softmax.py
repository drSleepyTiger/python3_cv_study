import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# tf Graph Input
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# Create model
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
activation = tf.nn.softmax(tf.matmul(X,W) + b)


# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(activation), reduction_indices=1))    #cross-entropy

learning_rate = tf.Variable(learning_rate)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

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
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys})/total_batch

        #Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", "%04d" % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # test model
    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(Y,1))

    # Calcuate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({X:mnist.test.images, Y: mnist.test.labels}))





