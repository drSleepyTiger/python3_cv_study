import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v0")

#input and output size base on the Env
input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

def one_hot(n, total = input_size):
    return np.identity(total)[n:n+1]

# Feed-forward part of the network used to choose action
X = tf.placeholder(shape=[1,input_size], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([input_size,output_size], 0, 0.01))

Qpred = tf.matmul(X,W)
Y = tf.placeholder(shape=[1,output_size], dtype=tf.float32)

loss = tf.reduce_sum(tf.square(Y-Qpred))

train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# set Q-learing related parameters
dis = .99
num_episodes = 2000

# Create lists to contain total rewards and steps per episode
rList = []

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    for i in range(num_episodes):
        # reset environment and get first new observation
        s = env.reset()

        # E&E(exploit&exploration) rate
        e = 1./((i/50)+10)      # episode 가 증가 할수록 e-greedy가 감소
        rAll = 0
        done = False
        local_loss = []

        while not done:
            # Choose an action
            Qs = sess.run(Qpred, feed_dict={X:one_hot(s)})          # Network 에 넣어서 해당 state(s)에 대한 output 계산

            if np.random.rand(1) < e:
                a = env.action_space.sample()   # take a random action

            else:
                a = np.argmax(Qs)               # 가장 높은 값으로 행동 함.

            # Get new state and reward from environment

            s1, reward, done, _ = env.step(a)
            if done:                      # Goal에 도달하였으면,
                Qs[0,a] = reward          # 해당 state 값의 그 action 값을 1 로 바꾸어줌

            else:                         # Goal이 아니면
                                          # 다음 스테이트에서 다시 Qs 계산하고 그 값중 가장 큰 값이 discount 를 적용한 값으로
                                          # 바꾸어 줌
                # Obtain the Q_s1 value by feeding the new state through our network
                Qs1 = sess.run(Qpred, feed_dict={X: one_hot(s1)})
                # update Q
                Qs[0,a] = reward + dis * np.max(Qs1)

            sess.run(train, feed_dict={X: one_hot(s), Y:Qs})

            rAll += reward
            s = s1

        # print("Episode: {0:4}, state: {1:2}, Result: {2}".format(i, s1,reward,done == 1.0))

        rList.append(rAll)
        if i % 100 == 0 and i > 0:
            print("Percent of successful episodes ({}~{}): {:.2f}%".format(i-100, i, 100 * sum(rList[-100:]) / len(rList[-100:])))

print("\nPercent of successful episodes: " + str(sum(rList)/num_episodes) + "%")
plt.bar(range(len(rList)), rList, color = "blue")
plt.show()