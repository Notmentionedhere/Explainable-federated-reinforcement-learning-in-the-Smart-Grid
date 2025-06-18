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
        self.num_actions = 25 # 5^N actions
        self.steps = 0

    def _explorationRate(self):
        # calculate decaying exploration rate
        if self.total_train_steps < self.exp_decay_steps:
            return self.exp_rate_start - self.total_train_steps * \
                   (self.exp_rate_start - self.exp_rate_end) / self.exp_decay_steps
        else:
            return self.exp_rate_end

    def step(self, exploration_rate, predict_net):
        # exploration rate determines the probability of random moves
        move = [0, -0.05, -0.1, 0.05, 0.1]
        if np.random.rand() < exploration_rate:
            action = np.random.randint(self.num_actions)

            # state_alpha, state_beta = self.env.getState()
            # if self.steps <= 11:
            #     if (state_alpha[0, 2] - move[2]) <= 1.01:
            #         if (state_beta[0, 2] - move[2]) <= 1:
            #             action = 2 * 5 + 2
            #         elif (state_beta[0, 2] - move[1]) <= 1:
            #             action = 2 * 5 + 1
            #         else:
            #             action = 2 * 5 + 0
            #     elif (state_alpha[0, 2] - move[1]) <= 1:
            #         if (state_beta[0, 2] - move[2]) <= 1:
            #             action = 1 * 5 + 2
            #         elif (state_beta[0, 2] - move[1]) <= 1:
            #             action = 1 * 5 + 1
            #         else:
            #             action = 1 * 5 + 0
            #     else:
            #         if (state_beta[0, 2] - move[2]) <= 1:
            #             action = 0 * 5 + 2
            #         elif (state_beta[0, 2] - move[1]) <= 1:
            #             action = 0 * 5 + 1
            #         else:
            #             action = 0 * 5 + 0
            # else:
            #     if (state_alpha[0, 2] - move[4]) >= 0:
            #         if (state_beta[0, 2] - move[4]) >= 0:
            #             action = 4 * 5 + 4
            #         elif (state_beta[0, 2] - move[3]) >= 0:
            #             action = 4 * 5 + 3
            #         else:
            #             action = 4 * 5 + 0
            #     elif (state_alpha[0, 2] - move[3]) >= 0:
            #         if (state_beta[0, 2] - move[4]) >= 0:
            #             action = 3 * 5 + 4
            #         elif (state_beta[0, 2] - move[3]) >= 0:
            #             action = 3 * 5 + 3
            #         else:
            #             action = 3 * 5 + 0
            #     else:
            #         if (state_beta[0, 2] - move[4]) >= 0:
            #             action = 0 * 5 + 4
            #         elif (state_beta[0, 2] - move[3]) >= 0:
            #             action = 0 * 5 + 3
            #         else:
            #             action = 0 * 5 + 0

            state_alpha, state_beta = self.env.getState()
            while True:
                action = np.random.randint(self.num_actions)
                act_a, act_b = divmod(action, 5)
                soc_a = state_alpha[0, 2] - move[act_a]
                soc_b = state_beta[0, 2] - move[act_b]
                if (-0.001 <= soc_a <= 1.001) and (-0.001 <= soc_b <= 1.001):
                    break

        else:
            # otherwise choose action with highest Q-value
            state_alpha, state_beta = self.env.getState()
            qvalue = self.net.predict(state_alpha, state_beta, predict_net)
            action = np.argmax(qvalue)
            while True:
                action = np.argmax(qvalue)
                act_a, act_b = divmod(action, 5)
                soc_a = state_alpha[0, 2] - move[act_a]
                soc_b = state_beta[0, 2] - move[act_b]
                if (-0.001 <= soc_a <= 1.001) and (-0.001 <= soc_b <= 1.001):
                    break
                else:
                    qvalue[action] = -100000

        # perform the action
        self.steps += 1
        #reward, action_rev = self.env.act(action, self.steps)
        reward = self.env.act(action, self.steps)
        state_alpha, state_beta = self.env.getState()   # state_all = [ ] state_all=self.env.getState()
        terminal = self.env.isTerminal()

        #return action_rev, reward, state_alpha, state_beta, terminal
        return action, reward, state_alpha, state_beta, terminal

    def train(self, epoch, train_episodes, outfile, predict_net):
        ep_loss, ep_rewards, details = [], [], []
        #min_samples = self.mem.batch_size + self.mem.hist_len
        min_samples = self.mem.batch_size
        # ipdb.set_trace()
        print('\n\n Training [%s] predicting [%s] ...' % (self.net.args.train_mode, predict_net))
        self.env.restart(data_flag='train', init=True)
        for episodes in range(train_episodes):
            self.steps = 0
            terminal = False
            while not terminal:
                act, r, s_a, s_b, terminal = self.step(self._explorationRate(), predict_net)
                self.mem.add(act, r, s_a, s_b, terminal)
                # Update target network every target_steps steps
                if self.target_steps and self.total_train_steps % self.target_steps == 0:
                    self.net.update_target_network()

                # train after every train_frequency steps
                if self.mem.count > min_samples and self.total_train_steps % self.train_frequency == 0:
                    # sample minibatch
                    minibatch = self.mem.getMinibatch()
                    # train the network
                    delta, loss = self.net.train(minibatch)
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
            self.env.restart(data_flag='train')

        # self.env.last_train_dom = self.env.dom_ind  # record last training domain

        return details

    def test(self, epoch, test_epidodes, outfile, predict_net, data_flag):
        success = 0.0
        min_steps = 0.0
        real_steps = 0.0
        test_reward = 0.0
        avg_reward = 0.0
        log_step_success = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}

        print('\n\n %s %s net ...' % (data_flag, predict_net))
        outfile.write('\n\n %s %s net ...\n' % (data_flag, predict_net))
        self.steps = 0
        self.env.restart(data_flag=data_flag, init=True)
        for ep in tqdm(range(test_epidodes)):
            terminal = False
            ep_rewards = []

            soc_a_record = np.zeros([1, 24], dtype=np.float16)
            soc_b_record = np.zeros([1, 24], dtype=np.float16)
            count1 = 0
            while not terminal:
                act, r, s_a, s_b, terminal = self.step(self.exploration_rate_test, predict_net)
                ep_rewards.append(r)

                soc_a = s_a[0, 2]
                soc_b = s_b[0, 2]
                soc_a_record[0, count1] = soc_a
                soc_b_record[0, count1] = soc_b
                count1 += 1
            if ep == 0:
                print(soc_a_record)
                print(soc_b_record)

            cum_reward = sum(ep_rewards)
            test_reward += cum_reward
            if self.env.episode_reward[-1][-1] <= 1:  # distance = 0 or 1 means that alpha meets beta
                success += 1.0
                min_steps += self.env.min_steps
                real_steps += self.steps
                for k in log_step_success:
                    if self.steps - self.env.min_steps <= k * self.env.min_steps:
                        log_step_success[k] += 1.0

            self.steps = 0
            self.env.restart(data_flag=data_flag)

        success_rate = success / test_epidodes
        avg_reward = test_reward / test_epidodes
        avg_steps = real_steps / success
        step_diff = (real_steps - min_steps) / min_steps
        for k in log_step_success:
            log_step_success[k] = log_step_success[k] / test_epidodes
        log_step_success[-1] = success_rate

        print('\n epochs: {}\t avg_reward: {:.2f}\t avg_steps: {:.2f}\t step_diff: {:.2f}'.format(epoch, avg_reward,
                                                                                                  avg_steps, step_diff))
        print('episodes: {}\t success_rate: {}\n'.format(test_epidodes, log_step_success))
        outfile.write('-----{}-----\n'.format(predict_net))
        outfile.write(
            '\n epochs: {}\t avg_reward: {:.2f}\t avg_steps: {:.2f}\t step_diff: {:.2f}\n'.format(epoch, avg_reward,
                                                                                                  avg_steps, step_diff))
        outfile.write('episodes: {}\t success_rate: {}\n\n'.format(test_epidodes, log_step_success))

        return log_step_success, avg_reward, step_diff, soc_a_record, soc_b_record