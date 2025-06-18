import numpy as np
import scipy.io as sio

class Environment(object):
    """ each state is a 0-1 matrix,
        where 0 denotes obstacle, 1 denotes space"""
    def __init__(self, args):
        self.args = args
        self.hist_len = args.hist_len   # 4
        self.state_dim = args.state_dim
        # self.image_dim = args.image_dim # 32
        # self.state_beta_dim = args.state_dim # 3
        # self.image_padding = args.image_padding
        # self.max_train_doms = args.max_train_doms       # 6400
        # self.start_valid_dom = args.start_valid_dom     # 6400
        # self.start_test_dom = args.start_test_dom       # 7200
        # self.step_reward = args.step_reward
        # self.collision_reward = args.collision_reward
        # self.terminal_reward = args.terminal_reward
        # self.move = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]]) # North, South, West, East
        self.move = [0, -0.05, -0.1, 0.05, 0.1] # + discharging -load
        self.soc_max = [25, 24, 32, 24, 32]
        self.baseload = 0.0228 #MVA
        # self.last_train_dom = -1
        # self.border_start = self.image_padding + 1  # >= 1
        # self.border_end = self.image_dim + self.image_padding - 2  # <= dim + pad - 2
        # self.padded_state_shape = (self.image_dim + self.image_padding*2, self.image_dim + self.image_padding*2)
        # self.state_alpha_dim = self.state_beta_dim + self.image_padding * 2
        # self.pos_bias = np.array([self.image_padding, self.image_padding])
        self.max_steps = 24 #8759
        self.num_agents = args.num_agents #4
        #self.load_data()

        self.states_all_train = np.zeros([self.num_agents, 1, self.state_dim], dtype=np.float16)
        self.soc_all = np.zeros([self.num_agents], dtype=np.float16)
        self.soc_all_train = np.zeros([self.num_agents], dtype=np.float16)
        self.dom_train = 0

    def load_data(self):
        # reward_mat = sio.loadmat('reward3.mat')
        # self.reward_mat = reward_mat["reward"]

        states = sio.loadmat('pvloaddata.mat') # 8759 * 2, 0:pv, 1:load
        self.states = states["pvloaddata"]
        #print(reward_mat[8759][0])

    def restart(self, data_flag, init=False):

        if init:
            if data_flag == 'train':  # training
                self.dom = 210
                self.dom_train = self.dom
                self.data_type = 'train'

            elif data_flag == 'valid':  # validation
                self.dom = 210 #3+16
                self.data_type = 'test'

            else:  # testing
                self.dom = 210 #3+16
                self.data_type = 'test'
            self.states_all = np.zeros([self.num_agents, 1, self.state_dim], dtype=np.float16)
            self.soc_all = np.zeros([self.num_agents], dtype=np.float16)
            for i in range(self.num_agents):
                self.soc_all[i] = 0.3
        else:
            if data_flag == 'train':
                self.data_type = 'train'
                self.dom = self.dom_train
                self.states_all = self.states_all_train
                self.soc_all = self.soc_all_train

        # for k in range(self.num_agent):
        #     exec(f'self.soc_{k} = 0.3')
        # self.soc_all = np.zeros([self.num_agents], dtype=np.float16)

        # for i in range(self.num_agents):
        #     self.soc_all[i] = 0.3

        starting_point = self.dom * self.max_steps + 19

        self.terminal = False
        self.episode_reward = []

        # self.states_all = np.zeros([self.num_agents, 1, self.state_dim], dtype=np.float16)
        for i in range(self.num_agents):
            #self.states_all[i][0] = [self.states[starting_point][0], self.states[starting_point][1], self.soc_all[i], 0.065*10, 24]
            self.states_all[i][0] = [self.soc_all[i], 0.065 * 10, 24/10]

        self.min_steps = 1


    def is_valid_soc(self, soc):
        # not in the border
        #return not (xy[0] >= self.image_dim-1 or xy[1] >= self.image_dim-1)
        #return not (soc + self.action > 1 or soc + self.action < 0)
        return not (soc > 1.01 or soc < -0.01)

    def act(self, action, steps):
        # act_a, act_b = divmod(action, 5) # 5^ N actions
        # new_soc_a = self.soc_a - self.move[act_a]
        # new_soc_b = self.soc_b - self.move[act_b]

        #act_all = np.zeros([self.num_agents], dtype=np.uint8)
        new_soc = np.zeros([self.num_agents], dtype=np.float16)
        # for i in range(self.num_agents):
        #     action, act_all[-(i+1)] = divmod(action, 5)  # 5^ N actions

        act_all = action
        for i in range(self.num_agents):
            new_soc[i] = self.soc_all[i] - self.move[act_all[i]]
        r_soc = 0
        for i in range(self.num_agents):
            if self.is_valid_soc(new_soc[i]):
                # soc not exceed limit
                self.soc_all[i] = new_soc[i]
            else:
                print('Invalid SoC')
                r_soc = r_soc - 10000.0
            #self.soc_all[i] = new_soc[i]

        time_step = (steps - 1 + 19) % 24  # 0~23
        if (time_step < 7) or (time_step >= 19):
            eprice = 0.065
        elif (time_step >= 11) and (time_step < 17):
            eprice = 0.132
        else: # 7~10 , 17^18
            eprice = 0.095

        starting_point = self.dom * self.max_steps + 19


        for i in range(self.num_agents):
            soc_max1 = self.soc_max[i % 5]
            if self.move[act_all[i]] > 0:
                alpha = 0.92
            else:
                alpha = 1 / 0.92
            # load_eff = self.states[starting_point+steps-1, 1] - 0.78*self.states[starting_point+steps-1, 0] \
            #     - (soc_max1 * 4 / 1000 * 70) * self.move[act_all[i]] * alpha
            load_eff = - (soc_max1 * 4 / 1000 * 70) * self.move[act_all[i]] * alpha
            r_soc += -load_eff * self.baseload * 1000 * eprice

        # compute reward
        #reward = r_soc + self.reward_mat[starting_point+steps-1][action1]
        reward = r_soc/100

        self.episode_reward.append([r_soc, reward])
        # add current state to states history
        time_step = (steps + 19) % 24  # 0~23
        if (time_step < 7) or (time_step >= 19):
            eprice = 0.065
        elif (time_step >= 11) and (time_step < 17):
            eprice = 0.132
        else:  # 7~10 , 17^18
            eprice = 0.095
        # self.states_alpha[0] = [self.states[starting_point+steps, 0], self.states[starting_point+steps, 1], self.soc_a, steps % 24] #states_alpha = np.zeros([1, 4, 2], dtype=np.float32)
        # self.states_beta[0] = [self.states[starting_point+steps, 0], self.states[starting_point+steps, 1], self.soc_b, steps % 24]
        for i in range(self.num_agents):
            # self.states_all[i][0] = [self.states[starting_point+steps, 0], self.states[starting_point+steps, 1], self.soc_all[i], eprice*10, 24 - (steps + 0) % 24]
            self.states_all[i][0] = [self.soc_all[i], eprice * 10, (24 - (steps + 0) % 24)/10]

        # terminal # distance = 0 or 1 means that alpha meets beta
        # if manhattan_distance <= 1 or steps >= self.max_steps:
        #     self.terminal = True
        # else:
        #     self.terminal = False
        if steps >= self.max_steps:
            # if self.data_type == 'train':
            #     self.dom += 1
            #     if ((self.dom + 1) % 4) == 0:
            #         self.dom += 1
            # else:
            #     self.dom += 4
            self.dom += 1
            if self.data_type == 'train':
                self.dom_train = self.dom
                self.states_all_train = self.states_all
                self.soc_all_train = self.soc_all
            self.terminal = True
        else:
            self.terminal = False

        return reward

    def getState(self):
        return self.states_all


    def isTerminal(self):
        return self.terminal



