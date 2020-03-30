import random
import numpy as np

# MountainCar-v0環境の構築
import gym
env = gym.make('MountainCar-v0')

# 深層強化学習環境の構築
from keras.models import Sequential
from keras.layers import Dense, Activation
model = Sequential()
model.add(Dense(3, input_dim = 2, activation='sigmoid'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(3))
model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['categorical_accuracy'])

EPOCH = 2000
ITER = 200
EPSIL = 0.1

def get_next_action_max_arg(model,observation):
    x = np.array([observation])
    reward = model.predict(x)
    a_max = np.argmax(reward)
    return a_max

def get_next_action_max_value(model,observation):
    x = np.array([observation])
    reward = model.predict(x)
    return np.amax(reward)

for i in range(EPOCH):

    #エージェントの初期化
    observation = env.reset()
    env.render()
    done = False

    #進行度を表示
    if i % 100 == 0:
        print(str((i/EPOCH)*100)+"% finished")

    # 0:左 1:そのまま　2:右
    action = env.action_space.n

    while not done:

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

        ############################

        ####  報酬から評価  ####

        label = model.predict(np.array([observation_now]))

        next_max = get_next_action_max_value(model,observation)
        q_value = reward + next_max
        label[0,next_action] = q_value

        x_data = np.array([observation_now])
        label = np.array(label)
        model.fit(x=x_data,y=label,verbose=0)
        ######################
