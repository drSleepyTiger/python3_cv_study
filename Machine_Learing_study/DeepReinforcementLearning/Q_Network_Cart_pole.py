import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")

#input and output size base on the Env
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
learning_rate = 0.1

# print(input_size, output_size)
#
# def one_hot(n, total = input_size):
#     return np.identity(total)[n:n+1]

# Feed-forward part of the network used to choose action
X = tf.placeholder(shape=[None,input_size], dtype=tf.float32, name = "input_x")

W1 = tf.get_variable("W1", shape=[input_size,output_size], initializer = tf.contrib.layers.xavier_initializer())

Qpred = tf.matmul(X,W1)
Y = tf.placeholder(shape=[None,output_size], dtype=tf.float32)

loss = tf.reduce_sum(tf.square(Y-Qpred))

train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# set Q-learing related parameters
dis = .99
num_episodes = 5000

# Create lists to contain total rewards and steps per episode
rList = []

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    for i in range(num_episodes):
        # reset environment and get first new observation
        s = env.reset()

        # E&E(exploit&exploration) rate
        e = 1./((i/10)+1)      # episode 가 증가 할수록 e-greedy가 감소
        rAll = 0
        step_count = 0
        done = False
        local_loss = []

        while not done:
            step_count += 1
            # Choose an action
            x = np.reshape(s,[1,input_size])

            Qs = sess.run(Qpred, feed_dict={X:x})          # Network 에 넣어서 해당 state(s)에 대한 output 계산

            if np.random.rand(1) < e:
                a = env.action_space.sample()   # take a random action

            else:
                a = np.argmax(Qs)               # 가장 높은 값으로 행동 함.

            # Get new state and reward from environment

            s1, reward, done, _ = env.step(a)
            x1 = np.reshape(s1,[1,input_size])

            if done:                      # Goal에 도달하였으면,
                Qs[0,a] = -100            # 해당 state 값의 그 action 값을 1 로 바꾸어줌

            else:                         # Goal이 아니면
                                          # 다음 스테이트에서 다시 Qs 계산하고 그 값중 가장 큰 값이 discount 를 적용한 값으로
                                          # 바꾸어 줌
                # Obtain the Q_s1 value by feeding the new state through our network
                Qs1 = sess.run(Qpred, feed_dict={X: x1})
                # update Q
                Qs[0,a] = reward + dis * np.max(Qs1)

            sess.run(train, feed_dict={X: x, Y:Qs})

            s = s1

        # print("Episode: {0:4}, state: {1:2}, Result: {2}".format(i, s1,reward,done == 1.0))

        rList.append(step_count)
        print("Episode: {} steps: {}".format(i, step_count))

        if len(rList) > 10 and np.mean(rList[-10:]) > 500:
            break

# see our trained network in action

observation = env.reset()
reward_sum = 0
while True:
    env.render()

    x = np.reshape(observation, [1,input_size])
    Qs = sess.run(Qpred, feed_dict={X:x})
    a = np.argmax(Qs)

    observation, reward, done, _ = env.step(a)
    reward_sum += reward

    if done:
        print("Total score: {}".format(reward_sum))
        break
