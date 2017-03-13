import gym
import numpy as np
import tensorflow as tf
import random
from collections import deque
import dqn
import time

env = gym.make("CartPole-v0")

#input and output size base on the Env
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
learning_rate = 0.05
dis = .9
REPLAY_MEMORY = 50000


# class DQN:
#
#     def __init__(self, session, input_size, output_size, name = "main"):
#         self.session = session
#         self.input_size = input_size
#         self.output_size = output_size
#         self.net_name = name
#
#         self._build_network()
#
#     def _build_network(self, h_size = 10, l_rate = 0.01):
#         with tf.variable_scope(self.net_name):
#             self._X = tf.placeholder(dtype = tf.float32, shape = [None, self.input_size], name = "input_x")
#
#             #First layer of weights
#             W1 = tf.get_variable("W1", shape=[self.input_size, h_size],initializer=tf.contrib.layers.xavier_initializer())
#             layer1 = tf.nn.tanh(tf.matmul(self._X, W1))
#
#             #Second layer of weights
#             W2 = tf.get_variable("W2", shape=[h_size,self.output_size],initializer=tf.contrib.layers.xavier_initializer())
#
#             #Q prediction
#             self._Qpred = tf.matmul(layer1, W2)
#
#         self._Y = tf.placeholder(dtype = tf.float32, shape = [None, self.output_size], name = "input_x")
#
#         # loss function
#         self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
#         # learning
#         self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)
#
#     def predict(self, state):
#         x = np.reshape(state, [1, self.input_size])
#         return self.session.run(self._Qpred, feed_dict = {self._X:x})
#
#     def update(self, x_stack, y_stack):
#         return self.session.run([self._loss, self._train], feed_dict = {self._X: x_stack, self._Y: y_stack})


def simple_replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0,mainDQN.input_size)
    y_stack = np.empty(0).reshape(0,mainDQN.output_size)

    #Get stored information from buffer
    for state, action, reward, next_state, done in train_batch:

        Q = mainDQN.predict(state)
        # terminal?
        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))

        y_stack = np.vstack([y_stack,Q])
        x_stack = np.vstack([x_stack,state])

    # Train our network using target and predicted Q values on each episode

    return mainDQN.update(x_stack, y_stack)

def bot_play(mainDQN):

    s = env.reset()
    reward_sum = 0
    done = False
    i = 0
    while not done:
        env.render()
        time.sleep(0.1)
        a= np.argmax(mainDQN.predict(s))
        s, reward, done, x = env.step(a)
        i += 1
        reward_sum += reward
        print(i, a, reward, done, x)

        if done:
            print("Total score: {}".format(reward_sum))


def get_copy_var_ops(dest_scope_name='target', src_scope_name='main'):

    op_holder = []

    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)

    for src_vars, dest_vars in zip(src_vars, dest_vars):
        op_holder.append(dest_vars.assign(src_vars.value()))

    return op_holder

def main():
    max_episodes = 20000

    # store the previous observation in replay memory
    replay_buffer = deque()
    rList = []

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess,input_size,output_size, l_rate = learning_rate, name = 'main')
        targetDQN = dqn.DQN(sess,input_size,output_size, l_rate = learning_rate, name = 'target')

        tf.global_variables_initializer().run()

        copy_ops = get_copy_var_ops(dest_scope_name='target', src_scope_name='main')


        for i in range(max_episodes):
            # reset environment and get first new observation
            state = env.reset()
            e = 1./((i/10)+1)           # E&E(exploit&exploration) rate
            step_count = 0
            explore_count = 0
            done = False

            while not done:

                # env.render()

                step_count += 1

                # if np.random.rand(1) < e:                                   # decaying E-greedy
                #     action = env.action_space.sample()   # take a random action
                #     explore_count += 1
                # else:
                #     action = np.argmax(mainDQN.predict(state))               # 가장 높은 값으로 행동 함.

                action = np.argmax(mainDQN.predict(state) + np.random.randn(1,2))

                next_state, reward, done, _ = env.step(action)
                if done:
                    reward = -100

                    rList.append(0.)
                    # print (next_state, reward, done)

                replay_buffer.append([state,action,reward,next_state,done])
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state

                if step_count > 10000:
                    rList.append(1.)
                    break

            rate = sum(rList[-30:]) / min(30,len(rList))
            if rate > 0.95:
                print("train_stop!")
                break

            print("Episode: {0} steps: {1} explore:{2} rate:{3:.2f}%".format(i, step_count, explore_count,100*rate))

            if i % 10 == 1 and i > 100:
                for j in range(50):
                    # Minibatch works better
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = simple_replay_train(mainDQN, targetDQN, minibatch)

                print ("replay size:",len(replay_buffer),"Loss: ", loss)

                sess.run(copy_ops)

        bot_play(mainDQN)

        t = time.strftime("%d%b%Y_%H%M%S",time.localtime())
        mainDQN.save("./saver/Cart_Pole_Nature2015_{}.ckpt".format(t))



if __name__ == "__main__":
    main()

