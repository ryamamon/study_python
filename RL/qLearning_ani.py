import numpy as np
import matplotlib.pyplot as plt
import random
import copy
#報酬値
MAZE =[[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
       [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1],
       [-1,-1,-1,-1,-1, 0,-1,-1, 0,-1],
       [-1,-1, 0, 0, 0, 0, 0,-1, 0,-1],
       [-1, 0,-1,-1,-1,-1,-1,-1, 0,-1],
       [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1],
       [-1, 0,-1,-1,-1,-1,-1,-1, 0,-1],
       [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1],
       [-1,-1, 0,-1,-1,-1,-1,-1,-1,-1],
       [-1,-1, 0, 0, 0, 0, 0, 0, 1,-1],
       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]]

ACTION = [[-1, 0], [1, 0], [0, -1], [0, 1]] # [上, 下, 左 , 右]
START = [1,1]
GOAL = [len(MAZE)-2, len(MAZE[0])-2]
EPOCH = 1000
ALPH = 0.1
GAMMA = 0.8
EPSIL = 0.1
RESULT = []
Q = np.random.randn(len(MAZE),len(MAZE[0]))

path = '/Users/ryota/Desktop/action.txt'
f = open(path,mode='w')

def state_check(state):
    if state[0] < 0 or state[1] < 0 or len(MAZE)-1 < state[0] or len(MAZE[0])-1 < state[1] :
        return 0
    return 1

#alpha: どれだけ元の値に影響を与えるか　　（学習率）
#gamma: 未来のことなので影響を少し弱める　（報酬割引率）
def update_q(Q,r,s,s_n,action,alpha,gamma):
    nextmax = get_next_max_value(Q,s_n,action)
    Q[s[0]][s[1]] = (1-alpha)*Q[s[0]][s[1]] + alpha*(r+gamma*nextmax)
    return Q

def get_next_max_value(Q,s_n,action):
    value_list = []
    for a in action:
        s_n_n = [0,0]
        s_n_n[0] = s_n[0] + a[0]
        s_n_n[1] = s_n[1] + a[1]

        # 移動先が枠外のとき
        if state_check(s_n_n) == 0:
            continue

        s_n_q = Q[s_n_n[0]][s_n_n[1]]
        value_list.append(s_n_q)

    return max(value_list)

def get_next_action_random(action):
    return random.choice(action)

def get_next_action_max_value(Q,s,action):
    value_list = []
    action_list = []
    for a in action:
        s_n = [0,0]
        s_n[0] = s[0] + a[0]
        s_n[1] = s[1] + a[1]

        # 移動先が枠外のとき
        if state_check(s_n) == 0:
            continue

        s_n_q = Q[s_n[0]][s_n[1]]
        value_list.append(s_n_q)
        action_list.append(a)

    max_index = value_list.index(max(value_list))
    return action_list[max_index]

def go_q(epoch=1000):
    q = Q
    action = ACTION
    for i in range(epoch):

        #進行度を表示
        if i % 100 == 0:
            print(str((i/epoch)*100)+"% finished")
            route = []

        #スタート地点
        state = copy.copy(START)

        ## アニメーション用　##
        f.write('0,0,3\n')
        ##################

        while state != GOAL:
            #移動先を選択
            rand = random.random()
            if rand < EPSIL:
                next_action = get_next_action_random(action)
            else:
                next_action = get_next_action_max_value(q,state,action)

            ## アニメーション用　##
            n_a_str = str(next_action[0])+','+str(next_action[1])+',0\n'
            f.write(n_a_str)
            ####################

            #移動先
            s_n = [0,0]
            s_n[0] = state[0] + next_action[0]
            s_n[1] = state[1] + next_action[1]

            #移動先が枠外ならば終了
            if state_check(s_n) == 0:
                break

            #移動先の報酬を取得
            r = MAZE[s_n[0]][s_n[1]]

            #Qを更新
            q = update_q(q,r,state,s_n,action,ALPH,GAMMA)

            #現在の場所の更新
            state = copy.copy(s_n)

            """""""""""""""""
            100回ごとにそのルートを表示
            """""""""""""""""
            #if i % 100 == 0:
                #route.append(copy.copy(state))

        #print(state)
        if state == GOAL:
            RESULT.append(1)
            ## アニメーション用　##
            f.write('0,1,2\n')
            ###################
        else:
            RESULT.append(0)

        """""""""""""""""
        100回ごとにそのルートを表示
        """""""""""""""""
        #if i % 100 == 0:
            #print(route)

def graph(result, epoch):
    x = np.arange(0, epoch, 1)
    left = np.array(x)
    count = 0
    count_1 = 0
    parcent = []
    for i in result:
        count += 1
        if i == 1:
            count_1 += 1
        parcent.append(count_1/count)
    height = np.array(parcent)
    plt.plot(left, height)
    plt.show()

go_q(EPOCH)
graph(RESULT,EPOCH)
