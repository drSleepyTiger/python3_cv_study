import gym
import numpy as np
import tensorflow as tf
import random
from collections import deque
import matplotlib.pyplot as plt
import dqn

env = gym.make("CartPole-v0")

#input and output size base on the Env
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
learning_rate = 0.01
dis = 0.9
REPLAY_MEMORY = 50000

def simple_replay_train(DQN, train_batch):
    x_stack = np.empty(0).reshape(0,DQN.input_size)
    y_stack = np.empty(0).reshape(0,DQN.output_size)

    #Get stored information from buffer
    for state, action, reward, next_state, done in train_batch:

        Q = DQN.predict(state)
        # terminal?
        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis * np.max(DQN.predict(next_state))

        y_stack = np.vstack([y_stack,Q])
        x_stack = np.vstack([x_stack,state])

    # np.set_printoptions(precision=3)
    # print ('y_stack:',y_stack)
    # Train our network using target and predicted Q values on each episode

    return DQN.update(x_stack, y_stack)

def bot_play(mainDQN):

    print("\n $$$$$$TEST$$$$$$$")
    s = env.reset()
    reward_sum = 0
    i = 0
    while True:
        env.render()
        a = np.argmax(mainDQN.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        print('step:', i, 'action', a, reward, done)
        if done:
            print("Total score: {}".format(reward_sum))
            break

def main():
    max_episodes = 50000

    # store the previous observation in replay memory
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess,input_size,output_size, h_size = 10, l_rate = learning_rate, name = 'main')
        tf.global_variables_initializer().run()

        for episode in range(max_episodes):
            # reset environment and get first new observation

            e = 1./((episode/10)+1)           # E&E(exploit&exploration) rate
            step_count = 0
            explore_count = 0
            done = False
            state = env.reset()

            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()   # take a random action
                    explore_count += 1
                else:
                    action = np.argmax(mainDQN.predict(state))               # 가장 높은 값으로 행동 함.

                next_state, reward, done, _ = env.step(action)
                if done:
                    reward = -100

                replay_buffer.append((state,action,reward,next_state,done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()
                    print ('memory full')
                #
                # if episode > 400:
                #     print ('step:',step_count, 'ene_cnt:',explore_count, 'action', action, reward, done)

                step_count += 1
                state = next_state
                if step_count > 10000:
                    break

            print("Episode: {} steps: {} e&e: {}".format(episode, step_count, explore_count))
            # if step_count > 10000:
            #     pass

            if episode % 10 == 1:
                print('buffer length:', len(replay_buffer))
                for j in range(50):
                    # Minibatch works better
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = simple_replay_train(mainDQN, minibatch)

                    # print(j,loss)

                print ("Loss: ", loss)

        bot_play(mainDQN)

if __name__ == "__main__":
    main()

