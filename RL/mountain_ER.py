import random
import numpy as np
from keras import backend as K
import tensorflow as tf
from collections import deque
import copy

"""
experience Replay
Fixed Target Q-Network
Reward Clipping
Huber Loss
"""
# huberloss (loss関数)
def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)

# MountainCar-v0環境の構築
import gym
env = gym.make('MountainCar-v0')

# 深層強化学習環境の構築

from keras.models import Sequential
from keras.layers import Dense, Activation
model = Sequential()
model.add(Dense(3, input_dim = 2, activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(3, activation='linear'))
model.compile(optimizer='Adam', loss=huberloss,metrics=['categorical_accuracy'])

EPOCH = 2000
ITER = 200
EPSIL = 0.1
gamma = 0.99
max_size = 2000
batch_size = 64

memory = deque(maxlen=max_size)

def get_next_action_max_arg(model,observation):
    x = np.array([observation])
    reward = model.predict(x)
    a_max = np.argmax(reward)
    return a_max

def get_next_action_max_value(model,observation):
    x = np.array([observation])
    reward = model.predict(x)
    return np.amax(reward)

def train_batch(mem,batch_size,model,target_model):

    return model

for i in range(EPOCH):

    #エージェントの初期化
    observation = env.reset()
    env.render()
    done = False
    target_model = model

    pos_lis = []

    #進行度を表示
    if i % 10 == 0:
        print(str(i)+"EPISODE finished")

    # 0:左 1:そのまま　2:右
    action = env.action_space.n

    while not done:

        pos_lis.append(observation[0])

        observation_now = observation

        #####　行動決定　####
        rand = random.random()
        if rand < EPSIL:
            next_action = env.action_space.sample()
        else:
            next_action = get_next_action_max_arg(model,observation)

        #####################

        ####  状態遷移、報酬取得  ####

        observation, reward, done, info = env.step(next_action)

        #### Reward Clipping ####

        ############################

        ####  報酬をメモリへ　####
        memory.append((observation_now,next_action,reward,observation))
        ####  パラメータ更新  ####

        if len(memory) >= batch_size:
            inputs = np.zeros((batch_size,2))
            targets = np.zeros((batch_size,3))
            #replace 重複
            rand = np.random.choice(np.arange(len(memory)),size=batch_size,replace=False)
            batch_train = [memory[i] for i in rand]
            for i, (state_b, action_b, reward_b, next_state_b) in enumerate(batch_train):
                inputs[i:i+1] = state_b
                targets[i] = model.predict(np.array([state_b]))
                n_a = get_next_action_max_arg(model,next_state_b)
                if next_state_b[0] >= 0.5:
                    q = reward_b
                else:
                    q = reward_b + gamma * target_model.predict(np.array([next_state_b]))[0][n_a]
                targets[i][action_b] = q
            #print(inputs)
            #print(targets)
            model.fit(inputs,targets,epochs=1,verbose=0)
        if target_model == model:
            print('True')
        target_model = model
        ######################

        if observation[0] >= 500:
            print(str(i)+"エピソード")
            print("登頂成功！！")

    print(max(pos_lis))
# env = wrappers.Monitor(env, './movie/cartpoleDDQN')  # 動画保存する場合
