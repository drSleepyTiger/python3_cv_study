import tensorflow as tf
import numpy as np

class DQN:

    def __init__(self, session, input_size, output_size, h_size = 10, l_rate = 0.01, name = "main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self.h_size = h_size
        self.l_rate = l_rate

        self._build_network()

    def _build_network(self):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(dtype = tf.float32, shape = [None, self.input_size], name = "input_x")

            #First layer of weights
            W1 = tf.get_variable("W1", shape=[self.input_size, self.h_size],initializer=tf.contrib.layers.xavier_initializer())
            layer1 = tf.nn.tanh(tf.matmul(self._X, W1))

            W2 = tf.get_variable("W2", shape=[self.h_size, self.h_size],initializer=tf.contrib.layers.xavier_initializer())
            layer2 = tf.nn.tanh(tf.matmul(layer1, W2))

            # Second layer of weights
            W3 = tf.get_variable("W3", shape=[self.h_size,self.output_size],initializer=tf.contrib.layers.xavier_initializer())
            self._Qpred = tf.matmul(layer2, W3)                 #Q prediction

            self.saver = tf.train.Saver()

        self._Y = tf.placeholder(dtype = tf.float32, shape = [None, self.output_size], name = "input_x")

        # loss function
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        # learning
        # self._train = tf.train.AdamOptimizer(learning_rate=self.l_rate).minimize(self._loss)
        self._train = tf.train.AdamOptimizer(learning_rate=self.l_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict = {self._X:x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict = {self._X: x_stack, self._Y: y_stack})

    def save(self,filename):
        save_path = self.saver.save(self.session, filename)
        print("Model saved in file:", save_path)
        return