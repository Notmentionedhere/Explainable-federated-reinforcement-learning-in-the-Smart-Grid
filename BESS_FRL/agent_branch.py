import numpy as np
from tqdm import tqdm


class Agent(object):
    """docstring for Agent"""

    def __init__(self, env, mem, dqn, args):
        self.env = env
        self.mem = mem
        self.net = dqn
 
 
        self.exp_rate_start = args.exploration_rate_start
        self.exp_rate_end = args.exploration_rate_end
        self.exp_decay_steps = args.exploration_decay_steps
        self.exploration_rate_test = args.exploration_rate_test
        self.total_train_steps = args.start_epoch * args.train_episodes * args.image_dim

        self.train_frequency = args.train_frequency
        self.target_steps = args.target_steps
        #self.num_actions = args.num_actions
        self.num_action = args.num_action # 5
        #self.num_actions = 5^4 # 5^N actions
        self.num_agents = args.num_agents
        self.test_episode = args.test_episodes
        self.steps = 0
        self.save_result = args.save_result
        self.result_record = np.zeros([3, 0], dtype=np.float16)
        self.result_1 = [[0], [0], [0]]
        self.num_agents = args.num_agents

    def _explorationRate(self):
        # calculate decaying exploration rate
        if self.total_train_steps < self.exp_decay_steps:
            return self.exp_rate_start - self.total_train_steps * \
                   (self.exp_rate_start - self.exp_rate_end) / self.exp_decay_steps
        else:
            return self.exp_rate_end

    def step(self, exploration_rate, predict_net, step0):
        # exploration rate determines the probability of random moves
        move = [0, -0.05, -0.1, 0.05, 0.1]
        if np.random.rand() < exploration_rate:
            #action = np.random.randint(self.num_actions)

            states_all = self.env.getState()
            act_all = np.zeros([self.num_agents], dtype=np.uint8)
            # new_soc = np.zeros([self.num_agents], dtype=np.float16)
            soc = np.zeros([self.num_agents], dtype=np.float16)
            for i in range(self.num_agents):
                #soc[i] = states_all[i, 0, 2]
                soc[i] = states_all[i, 0, 0]
            if np.random.rand() > exploration_rate:
                # for i in range(self.num_agents): #random
                #     while True:
                #         action = np.random.randint(self.num_action)
                #         new_soc = soc[i] - move[action]
                #         if -0.001 <= new_soc <= 1.001:
                #             act_all[i] = action
                #             break
                for i in range(self.num_agents): #random
                    if soc[i] > 0.98: #[0, -0.05, -0.1, 0.05, 0.1]
                        action_id = np.random.randint(self.num_action-2)
                        if action_id > 0:
                            action = action_id + 2
                        else:
                            action = action_id
                    elif soc[i] > 0.93: #[0, -0.05, -0.1, 0.05, 0.1]
                        action_id = np.random.randint(self.num_action-1)
                        if action_id > 1:
                            action = action_id + 1
                        else:
                            action = action_id
                    elif soc[i] < 0.02: #[0, -0.05, -0.1, 0.05, 0.1]
                        action_id = np.random.randint(self.num_action-2)
                        action = action_id
                    elif soc[i] < 0.06: #[0, -0.05, -0.1, 0.05, 0.1]
                        action_id = np.random.randint(self.num_action-1)
                        action = action_id
                    else:
                        action = np.random.randint(self.num_action)
                    act_all[i] = action

            else:
                time_step = (step0 + 19) % 24
                for i in range(self.num_agents): #blind
                    if (time_step < 7) or (time_step >= 19):
                        action = 2
                        new_soc = soc[i] - move[action]
                        if (new_soc > 1.001) or (new_soc < -0.001):
                            action = 1
                            new_soc = soc[i] - move[action]
                            if (new_soc > 1.001) or (new_soc < -0.001):
                                action = 0
                        act_all[i] = action
                    elif (time_step >= 11) and (time_step < 17):
                        action = 4
                        new_soc = soc[i] - move[action]
                        if (new_soc > 1.001) or (new_soc < -0.001):
                            action = 3
                            new_soc = soc[i] - move[action]
                            if (new_soc > 1.001) or (new_soc < -0.001):
                                action = 0
                        act_all[i] = action
                    else:
                        action = 3
                        new_soc = soc[i] - move[action]
                        if (new_soc > 1.001) or (new_soc < -0.001):
                            action = 0
                        act_all[i] = action


        else:
            # otherwise choose action with highest Q-value
            # state_alpha, state_beta = self.env.getState() # [1,4]
            # qvalue = self.net.predict(state_alpha, state_beta, predict_net)
            states_all = self.env.getState()  # [2,1,4]
            qvalue = self.net.predict(states_all, predict_net)

            # act_all = np.zeros([self.num_agents], dtype=np.int8)
            # for i in range(self.num_agents):
            #     act_all[i] = np.argmax(qvalue[i])
            # action = 0
            # for i in range(self.num_agents):
            #     action += act_all[i] * (5**(self.num_agents-i-1))

            act_all = np.zeros([self.num_agents], dtype=np.uint8)
            new_soc = np.zeros([self.num_agents], dtype=np.float16)
            soc = np.zeros([self.num_agents], dtype=np.float16)
            for i in range(self.num_agents):
                #soc[i] = states_all[i, 0, 2]
                soc[i] = states_all[i, 0, 0]

            # while True:
            #     counter_soc = 0
            #     for i in range(self.num_agents):
            #         act_all[i] = np.argmax(qvalue[i])
            #     for i in range(self.num_agents):
            #         new_soc[i] = soc[i] - move[act_all[i]]
            #     for i in range(self.num_agents):
            #         if (new_soc[i] <= -0.001) or (new_soc[i] >= 1.001):
            #             qvalue[i][act_all[i]] = -100000
            #         else:
            #             counter_soc += 1
            #     if counter_soc == self.num_agents:
            #         action = 0
            #         for i in range(self.num_agents):
            #             action += act_all[i] * (5 ** (self.num_agents - i - 1))
            #         break

            for i in range(self.num_agents):
                while True:
                    act_all[i] = np.argmax(qvalue[i])
                    new_soc[i] = soc[i] - move[act_all[i]]
                    if (new_soc[i] <= -0.001) or (new_soc[i] >= 1.001):
                        qvalue[i][act_all[i]] = -100000
                    else:
                        break
            # action = 0
            # for i in range(self.num_agents):
            #     action += act_all[i] * (5 ** (self.num_agents - i - 1))



        # perform the action
        self.steps += 1
        reward = self.env.act(act_all, self.steps)
        states_all = self.env.getState()   # state_all = [self.num_agent, 1, self.state_dim]
        terminal = self.env.isTerminal()

        return act_all, reward, states_all, terminal

    def train(self, epoch, train_episodes, predict_net):
        ep_loss, ep_rewards, details = [], [], []
        avg_loss = 0
        #min_samples = self.mem.batch_size + self.mem.hist_len
        min_samples = self.mem.batch_size
        # ipdb.set_trace()
        print('\n\n Training [%s] predicting [%s] ...' % (self.net.args.train_mode, predict_net))
        self.env.restart(data_flag='train', init=True)
        for episodes in range(train_episodes):
            self.steps = 0
            terminal = False
            while not terminal:
                # act, r, s_a, s_b, terminal = self.step(self._explorationRate(), predict_net)
                # self.mem.add(act, r, s_a, s_b, terminal)

                act, r, s_all, terminal = self.step(self._explorationRate(), predict_net, self.steps)
                self.mem.add(act, r, s_all, terminal)
                # Update target network every target_steps steps
                if self.target_steps and self.total_train_steps % self.target_steps == 0:
                    #print("loss1")
                    self.net.update_target_network()
                    #print("loss2")

                # train after every train_frequency steps
                if self.mem.count > min_samples and self.total_train_steps % self.train_frequency == 0:
                    # sample minibatch
                    minibatch = self.mem.getMinibatch()
                    # train the network
                    #print("loss1")
                    loss = self.net.train(minibatch)
                    #print("loss2")
                    ep_loss.append(loss)

                ep_rewards.append(r)
                self.total_train_steps += 1

            # print('domain: %d \t min_steps: %d \t max_steps: %d' % (
            # self.env.dom_ind, self.env.min_steps, self.env.max_steps))
            if len(ep_loss) > 0:
                avg_loss = sum(ep_loss) / len(ep_loss)
                max_loss = max(ep_loss)
                min_loss = min(ep_loss)
                print(
                    'max_loss: {:>6.6f}\t min_loss: {:>6.6f}\t avg_loss: {:>6.6f}'.format(max_loss, min_loss, avg_loss))

            cum_reward = sum(ep_rewards)
            details.append(self.env.episode_reward)
            print('epochs: {}\t episodes: {}\t steps: {}\t cum_reward: {:>6.6f}\n'.format(epoch, episodes, self.steps,
                                                                                          cum_reward))
            ep_loss, ep_rewards = [], []
            if self.save_result == 1:
                self.result_1[0][0] = avg_loss
                self.result_1[1][0] = cum_reward
                self.test(epoch, self.test_episode, 'both', 'valid', 1)

            self.env.restart(data_flag='train')

        # self.env.last_train_dom = self.env.dom_ind  # record last training domain

        return details

    def test(self, epoch, test_epidodes, predict_net, data_flag, writefile):
        success = 0.0
        min_steps = 0.0
        real_steps = 0.0
        test_reward = 0.0
        avg_reward = 0.0
        log_step_success = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}

        print('\n\n %s %s net ...' % (data_flag, predict_net))
        #outfile.write('\n\n %s %s net ...\n' % (data_flag, predict_net))
        self.steps = 0
        self.env.restart(data_flag=data_flag, init=True)
        count1 = 0
        soc_a_record = np.zeros([1, 48], dtype=np.float16)
        soc_b_record = np.zeros([1, 48], dtype=np.float16)
        for ep in tqdm(range(test_epidodes)):
            terminal = False
            ep_rewards = []

            while not terminal:
                #act, r, s_a, s_b, terminal = self.step(self.exploration_rate_test, predict_net)
                act, r, s_all, terminal = self.step(self.exploration_rate_test, predict_net, self.steps)
                ep_rewards.append(r)
                if ep == 0 or ep == 1:
                    # soc_a = s_all[0, 0, 2]
                    # soc_b = s_all[1, 0, 2]
                    soc_a = s_all[0, 0, 0]
                    soc_b = s_all[1, 0, 0]
                    soc_a_record[0, count1] = soc_a
                    soc_b_record[0, count1] = soc_b
                    count1 += 1
            if ep == 1 and writefile != 1:
                print(soc_a_record)
                print(soc_b_record)

            cum_reward = sum(ep_rewards)
            test_reward += cum_reward
            # if self.env.episode_reward[-1][-1] <= 1:  # distance = 0 or 1 means that alpha meets beta
            #     success += 1.0
            #     min_steps += self.env.min_steps
            #     real_steps += self.steps
            #     for k in log_step_success:
            #         if self.steps - self.env.min_steps <= k * self.env.min_steps:
            #             log_step_success[k] += 1.0

            self.steps = 0
            self.env.restart(data_flag='valid')

        #success_rate = success / test_epidodes
        avg_reward = test_reward / test_epidodes
        # avg_steps = real_steps / success
        # step_diff = (real_steps - min_steps) / min_steps
        # for k in log_step_success:
        #     log_step_success[k] = log_step_success[k] / test_epidodes
        # log_step_success[-1] = success_rate

        # print('\n epochs: {}\t avg_reward: {:.2f}\t avg_steps: {:.2f}\t step_diff: {:.2f}'.format(epoch, avg_reward,
        #                                                                                           avg_steps, step_diff))
        print('\n epochs: {}\t avg_reward: {:.2f}\t'.format(epoch, avg_reward))
        #print('episodes: {}\t success_rate: {}\n'.format(test_epidodes, log_step_success))
        print('episodes: {}\t \n'.format(test_epidodes))
        # outfile.write('-----{}-----\n'.format(predict_net))
        # outfile.write(
        #     '\n epochs: {}\t avg_reward: {:.2f}\t avg_steps: {:.2f}\t step_diff: {:.2f}\n'.format(epoch, avg_reward,
        #                                                                                           avg_steps, step_diff))
        # outfile.write('episodes: {}\t success_rate: {}\n\n'.format(test_epidodes, log_step_success))
        if self.save_result == 1:
            if writefile == 1:
                self.result_1[2][0] = avg_reward
                self.result_record = np.append(self.result_record, self.result_1, axis=1)
                print(len(self.result_record[0]))
        #return log_step_success, avg_reward, step_diff, soc_a_record, soc_b_record
        return avg_reward, soc_a_record, soc_b_record

    def save_record(self):
        return self.result_record