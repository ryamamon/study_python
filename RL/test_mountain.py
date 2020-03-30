import random
import numpy as np
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
from collections import deque
import copy
import matplotlib.pyplot as plt
import time
import sys

"""
experience Replay　　　　　◎
Fixed Target Q-Network　　◎
Reward Clipping　　　　　　　
Huber Loss　　　　　　　　　◎
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

class Qnet:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(24, input_dim = 2, activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(3, activation='linear'))
        self.model.compile(optimizer=Adam(lr=0.001), loss=huberloss,metrics=['categorical_accuracy'])

    def copy_weights(self,model):
        model.save_weights('model_copy__weights.h5')
        self.model.load_weights('model_copy__weights.h5')

EPOCH = 4000
ITER = 200
EPSIL = 0.5
gamma = 0.99
alpha = 0.2
initialize_size = 2000 #この大きさを超えてから学習開始
max_size = 20000 #キューの大きさ
convert_size = 50 # ターゲットQの変更頻度
batch_size = 32


main = Qnet()
target = Qnet()

memory = deque(maxlen=max_size)
reward_list = []

def get_next_action_max_arg(model,observation):
    x = np.array([observation])
    reward = model.predict(x)
    a_max = np.argmax(reward)
    return a_max

def get_next_action_max_value(model,observation):
    x = np.array([observation])
    reward = model.predict(x)
    return np.amax(reward)

action = env.action_space.n
epoch = [0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500]
clear_flg = False
for i in epoch:
    main.model.load_weights('param_ok/model_'+str(i)+'_epi_24_24_24_fix.h5')
    observation = env.reset()
    done = False
    time.sleep(1)
    print(str(i)+ "epoch ")
    while not done:
        env.render()
        next_action = get_next_action_max_arg(main.model,observation)
        observation, reward, done, info = env.step(next_action)
        #if observation[0] >= 0.5:
            #if clear_flg:
                #sys.exit()
            #else:
                #clear_flg = True
