import os
import ipdb
import numpy as np
import tensorflow as tf
from utils import save_pkl, load_pkl
from tensorflow.contrib.layers.python.layers import initializers
from functools import reduce


class FRLDQN(object):
    """docstring for FRLNetwork"""

    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        self.gamma = args.gamma
        #self.gamma = 0.95
        self.lambda_ = args.lambda_
        self.preset_lambda = args.preset_lambda
        self.add_train_noise = args.add_train_noise
        self.add_predict_noise = args.add_predict_noise
        self.noise_prob = args.noise_prob
        self.stddev = args.stddev
        #self.num_actions = args.num_actions
        self.num_actions = args.num_actions
        self.agent_outlayer = 25
        self.num_agents = args.num_agents  # 4
        self.state_dim = args.state_dim
        self.learning_rate = args.learning_rate
        self.hist_len = args.hist_len
        self.state_beta_dim = args.state_dim
        self.state_alpha_dim = args.state_dim + args.image_padding * 2
        self.build_dqn()


    def conv2d(self, x, output_dim, kernel_size, stride, initializer, activation_fn=None, padding='VALID',
               name='conv2d'):
        with tf.variable_scope(name):
            # data_format = 'NHWC'
            stride = [1, stride[0], stride[1], 1]
            kernel_size = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

            w = tf.get_variable('w', kernel_size, tf.float32, initializer=initializer)
            conv = tf.nn.conv2d(x, w, stride, padding)

            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.1))
            out = tf.nn.bias_add(conv, b)

        if activation_fn != None:
            out = activation_fn(out)
        return out, w, b

    def max_pooling(self, x, kernel_size, stride, padding='VALID', name='max_pool'):
        with tf.variable_scope(name):
            stride = [1, stride[0], stride[1], 1]
            kernel_size = [1, kernel_size[0], kernel_size[1], 1]
            return tf.nn.max_pool(x, kernel_size, stride, padding)

    def linear(self, x, output_dim, activation_fn=None, name='linear'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [x.get_shape()[1], output_dim], tf.float32,
                                initializer=tf.truncated_normal_initializer(0, 0.1))
            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.1))
            out = tf.nn.bias_add(tf.matmul(x, w), b)

        if activation_fn != None:
            out = activation_fn(out)
        return out, w, b

    def build_dqn(self):
        # init = tf.contrib.layers.xavier_initializer()
        init = tf.contrib.layers.xavier_initializer_conv2d()
        summary = []

        def build_nn(name, weight, s_t, summary=summary):
            print(1)
            # fw = s_t.shape[2] if self.args.autofilter else 3
            with tf.variable_scope(name):
                print('Initializing %s network ...' % name)
                # l1, weight['l1_w'], weight['l1_b'] = self.conv2d(s_t, 32, [fw, fw], [1, 1], init, tf.nn.relu, 'SAME',
                #                                                  name='l1')
                l1, weight['l1_w'], weight['l1_b'] = self.linear(s_t, 32, tf.nn.relu, name='l1')
                #l1, weight['l1_w'], weight['l1_b'] = self.linear(s_t, 32, name='l1')
                # l1_shape = l1.get_shape().as_list()
                # l1_flat = tf.reshape(l1, [-1, reduce(lambda x, y: x * y, l1_shape[1:])])
                l2, weight['l2_w'], weight['l2_b'] = self.linear(l1, 256, tf.nn.relu, name='l2')
                #l2, weight['l2_w'], weight['l2_b'] = self.linear(l1, 256, name='l2')
                #out_layer, weight['q_w'], weight['q_b'] = self.linear(l2, self.num_actions, name='q')
                out_layer, weight['q_w'], weight['q_b'] = self.linear(l2, self.agent_outlayer, name='q') #TODO 5
                #summary += [l1, l1_flat, l2, out_layer, '']
                summary += [l1, l2, out_layer, '']
                return out_layer

        def build_nn_full(name, weight, s_a, s_b, summary=summary):
            # if self.args.autofilter:
            #     fw_a, fw_b = s_a.shape[2], s_b.shape[2]
            # else:
            #     fw_a = fw_b = 3
            print(2)
            with tf.variable_scope(name):
                print('Initializing %s network ...' % name)
                # l1, weight['l1_w'], weight['l1_b'] = self.conv2d(s_a, 32, [fw_a, fw_a], [1, 1], init, tf.nn.relu,
                #                                                  'SAME', name='l1')
                # l2, weight['l2_w'], weight['l2_b'] = self.conv2d(s_b, 32, [fw_b, fw_b], [1, 1], init, tf.nn.relu,
                #                                                  'SAME', name='l2')
                l1, weight['l1_w'], weight['l1_b'] = self.linear(s_a, 32, tf.nn.relu, name='l1')
                l2, weight['l2_w'], weight['l2_b'] = self.linear(s_b, 32, tf.nn.relu, name='l2')

                l1_shape = l1.get_shape().as_list()
                l1_flat = tf.reshape(l1, [-1, reduce(lambda x, y: x * y, l1_shape[1:])])
                l2_shape = l2.get_shape().as_list()
                l2_flat = tf.reshape(l2, [-1, reduce(lambda x, y: x * y, l2_shape[1:])])
                l3 = tf.concat([l1_flat, l2_flat], axis=1)
                l4, weight['l2_w'], weight['l2_b'] = self.linear(l3, 256, tf.nn.relu, name='l3')
                out_layer, weight['q_w'], weight['q_b'] = self.linear(l4, self.num_actions, name='q')

                summary += [l1, l2, l1_flat, l2_flat, l3, l4, out_layer, '']
                return out_layer

        #def build_mlp(name, weight, alpha_q, beta_q, summary=summary):
        def build_mlp(name, weight, agent_q, summary=summary):
            # MLP network for combining two Q-value tensors
            with tf.variable_scope(name):
                #hidden_size = 2 * self.num_actions   # equals to the dimension of the concated q-values
                #concat_q = tf.concat([alpha_q, beta_q], axis=1)
                concat_q = tf.concat(agent_q, axis=1)

                # hidden_layer, weight['mlp_h_w'], weight['mlp_h_b'] = self.linear(concat_q, hidden_size, tf.nn.relu,
                #                                                                 name='mlp_hidden')
                # hidden_layer, weight['mlp_h_w'], weight['mlp_h_b'] = self.linear(concat_q, hidden_size,
                #                                                                  name='mlp_hidden')
                mlp_output, weight['mlp_q_w'], weight['mlp_q_b'] = self.linear(concat_q, self.num_actions,
                                                                               name='mlp_q')

                # summary += [concat_q, hidden_layer, mlp_output, '']
                summary += [concat_q, mlp_output, '']
                return mlp_output

        # ipdb.set_trace()
        # construct DQN-beta network
        # self.beta_w, self.beta_t_w = {}, {}
        # self.s_b = tf.placeholder(tf.float32, [None, self.state_dim], 's_b')
        # self.beta_q = build_nn('beta_q', self.beta_w, self.s_b)
        # self.beta_t_q = build_nn('beta_t_q', self.beta_t_w, self.s_b)
        #
        # # construct DQN-alpha network
        # self.s_a = tf.placeholder(tf.float32, [None, self.state_dim], 's_a')
        # self.alpha_w, self.alpha_t_w = {}, {}
        # self.alpha_q = build_nn('alpha_q', self.alpha_w, self.s_a)
        # self.alpha_t_q = build_nn('alpha_t_q', self.alpha_t_w, self.s_a)

        self.agent_w, self.agent_t_w = [], []
        self.agent_q, self.agent_t_q = [], []
        self.s_all = []
        for i in range(self.num_agents):
            #s_int = tf.placeholder(tf.float32, [None, self.state_dim], 's_%s' % i)
            self.s_all.append(tf.placeholder(tf.float32, [None, self.state_dim], 's_%s' % i))
            weights_w, weights_t_w = {}, {}
            self.agent_w.append(weights_w)
            self.agent_t_w.append(weights_t_w)
            # agent_q = build_nn('agent_q_%s' % i, self.agent_w[i], self.s_all[i])
            # agent_t_q = build_nn('agent_t_q_%s' % i, self.agent_t_w[i], self.s_all[i])
            self.agent_q.append(build_nn('agent_q_%s' % i, self.agent_w[i], self.s_all[i]))
            self.agent_t_q.append(build_nn('agent_t_q_%s' % i, self.agent_t_w[i], self.s_all[i]))


        # construct DQN-full network
        # self.full_w, self.full_t_w = {}, {}
        # self.full_q = build_nn_full('full_q', self.full_w, self.s_a, self.s_b)
        # self.full_t_q = build_nn_full('full_t_q', self.full_t_w, self.s_a, self.s_b)

        # construct FRL network
        # self.frl_w, self.frl_t_w = {}, {}
        # for k, v in self.alpha_w.items():  # update all alpha weights to frl weights
        #     self.frl_w['alpha_' + k] = v
        # for k, v in self.beta_w.items():  # update all beta weights to frl weights
        #     self.frl_w['beta_' + k] = v
        # for k, v in self.alpha_t_w.items():
        #     self.frl_t_w['alpha_' + k] = v
        # for k, v in self.beta_t_w.items():
        #     self.frl_t_w['beta_' + k] = v

        self.frl_w, self.frl_t_w = {}, {}
        for i in range(self.num_agents):
            for k, v in self.agent_w[i].items():  # update all alpha weights to frl weights
                self.frl_w['%s_%s' % (i, k)] = v
            for k, v in self.agent_t_w[i].items():
                self.frl_t_w['%s_%s' % (i, k)] = v

        # prepare two place-holders for Q-alpha and Q-beta
        # self.alpha_q_input = tf.placeholder(tf.float32, [None, self.num_actions], 'alpha_q_input')
        # self.beta_q_input = tf.placeholder(tf.float32, [None, self.num_actions], 'beta_q_input')
        # self.frl_q = build_mlp('frl_q', self.frl_w, self.alpha_q_input, self.beta_q_input)
        # self.frl_t_q = build_mlp('frl_t_q', self.frl_t_w, self.alpha_q_input, self.beta_q_input)

        self.agent_q_input = []
        for i in range(self.num_agents):
            #self.agent_q_input.append(tf.placeholder(tf.float32, [None, self.state_dim], 's_%s' % i))
            #self.agent_q_input.append(tf.placeholder(tf.float32, [None, self.num_actions], '%s_q_input' % i))
            self.agent_q_input.append(tf.placeholder(tf.float32, [None, self.agent_outlayer], '%s_q_input' % i))
        # self.frl_q = build_mlp('frl_q', self.frl_w, self.agent_q_input[0], self.agent_q_input[1])
        # self.frl_t_q = build_mlp('frl_t_q', self.frl_t_w, self.agent_q_input[0], self.agent_q_input[1]) #TODO
        self.frl_q = build_mlp('frl_q', self.frl_w, self.agent_q_input)
        self.frl_t_q = build_mlp('frl_t_q', self.frl_t_w, self.agent_q_input)


        # print summary of all layers
        for layer in summary:
            try:
                print('{}\t{}'.format(layer.shape, layer))
            except:
                print('\n')

        # construct the update q-network operators
        # self.alpha_t_w_input, self.alpha_t_w_assign_op = self.update_q_network_op(self.alpha_t_w,
        #                                                                           'alpha_update_q_network_op')
        # self.beta_t_w_input, self.beta_t_w_assign_op = self.update_q_network_op(self.beta_t_w,
        #                                                                         'beta_update_q_network_op')
        # self.full_t_w_input, self.full_t_w_assign_op = self.update_q_network_op(self.full_t_w,
        #                                                                         'full_update_q_network_op')
        # self.frl_t_w_input, self.frl_t_w_assign_op = self.update_q_network_op(self.frl_t_w, 'frl_update_q_network_op')

        self.agent_t_w_input,  self.agent_t_w_assign_op = [], []
        for i in range(self.num_agents):
            agent_t_w_input, agent_t_w_assign_op = self.update_q_network_op(self.agent_t_w[i],
                                                                                      '%s_update_q_network_op' % i)
            self.agent_t_w_input.append(agent_t_w_input)
            self.agent_t_w_assign_op.append(agent_t_w_assign_op)
        self.frl_t_w_input, self.frl_t_w_assign_op = self.update_q_network_op(self.frl_t_w, 'frl_update_q_network_op') #TODO


        with tf.variable_scope('optimizer'):
            print('Initializing optimizer ...')
            self.target_q = tf.placeholder(tf.float32, [None, self.num_actions], 'targets')
            # self.delta_full = self.target_q - self.full_q
            # self.delta_beta = self.target_q - self.beta_q
            # self.delta_alpha = self.target_q - self.alpha_q
            if self.preset_lambda:
                # use preset lambda to control Q_alpha and Q_beta
                # self.delta_frl = self.target_q - self.lambda_ * self.alpha_q - (1 - self.lambda_) * self.beta_q

                self.delta_frl = self.target_q
                for i in range(self.num_agents):
                    self.delta_frl = self.delta_frl - self.agent_q[i] / self.num_agents
            else:
                # use MLP to automatically control Q_alpha and Q_beta
                self.delta_frl = tf.stop_gradient(self.target_q) - self.frl_q  # 25 actions' Q one - one; delta_frl (,16)

            # self.loss_full = tf.reduce_sum(tf.square(self.delta_full), name='loss_full')
            # self.loss_beta = tf.reduce_sum(tf.square(self.delta_beta), name='loss_beta')
            # self.loss_alpha = tf.reduce_sum(tf.square(self.delta_alpha), name='loss_alpha')
            #self.loss_frl = tf.reduce_sum(tf.square(self.delta_frl), name='loss_frl')  #sum(squares root) = 1 number
            self.epsilon = tf.constant([0.5])
            #self.loss_frl = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(self.delta_frl), tf.stop_gradient(self.epsilon))), name='loss_frl')
            self.loss_frl = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(self.delta_frl), self.epsilon)), name='loss_frl')

            # self.train_full_dqn = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_full)
            # self.train_single_beta = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_beta)
            # self.train_single_alpha = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_alpha)
            # experimentally we can train both alpha network and beta network at the same time
            self.train_frl = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_frl)

            # but in reality, we should train the two networks separatelly
            #self.mlp_weights, self.alpha_weights, self.beta_weights = [], [], []
            self.mlp_weights, self.agent_weights = [], []

            # self.frl_w, self.frl_t_w = {}, {}
            # for i in range(self.num_agents):
            #     for k, v in self.agent_w[i].items():  # update all alpha weights to frl weights
            #         self.frl_w['%s_%s' % (i, k)] = v
            #     for k, v in self.agent_t_w[i].items():
            #         self.frl_t_w['%s_%s' % (i, k)] = v
            for i in range(self.num_agents):
                # s_int = tf.placeholder(tf.float32, [None, self.state_dim], 's_%s' % i)
                #self.s_all.append(tf.placeholder(tf.float32, [None, self.state_dim], 's_%s' % i))
                agent_weights = []
                self.agent_weights.append(agent_weights)
                for name, tensor in self.frl_w.items():
                    if ('%s_l' % i in name) or ('%s_q' % i in name):
                        self.agent_weights[i].append(tensor)
                # agent_q = build_nn('agent_q_%s' % i, self.agent_w[i], self.s_all[i])
                # agent_t_q = build_nn('agent_t_q_%s' % i, self.agent_t_w[i], self.s_all[i])
                #self.agent_q.append(build_nn('agent_q_%s' % i, self.agent_w[i], self.s_all[i]))
            for name, tensor in self.frl_w.items():
                if 'mlp' in name:
                    self.mlp_weights.append(tensor)
            # for name, tensor in self.frl_w.items():
            #     if 'alpha' in name:
            #         self.alpha_weights.append(tensor)
            #     elif 'beta' in name:
            #         self.beta_weights.append(tensor)
            #     else:
            #         self.mlp_weights.append(tensor)

            # train mlp
            # ipdb.set_trace()
            if not self.preset_lambda: #TODO
                self.mlp_grads = tf.gradients(self.loss_frl, self.mlp_weights)
                self.train_mlp = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                    zip(self.mlp_grads, self.mlp_weights))
                # compute the gradients and pass them to alpha and beta network
                # self.dloss_dQa = tf.gradients(self.loss_frl, self.alpha_q_input)
                # self.dloss_dQb = tf.gradients(self.loss_frl, self.beta_q_input)

                self.dloss_dQ = []
                for i in range(self.num_agents):
                    self.dloss_dQ.append(tf.gradients(self.loss_frl, self.agent_q_input[i]))

                # # train beta net
                # self.dloss_dQb_input = tf.placeholder(tf.float32, [None, self.num_actions], 'dloss_dQb_input')
                # # self.beta_weights = [tensor for name, tensor in self.frl_w.items() if 'beta' in name]
                # self.beta_grads = tf.gradients(self.beta_q, self.beta_weights, self.dloss_dQb_input)
                # self.train_beta = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                #     zip(self.beta_grads, self.beta_weights))
                # # train alpha net
                # self.dloss_dQa_input = tf.placeholder(tf.float32, [None, self.num_actions], 'dloss_dQa_input')
                # # self.alpha_weights = [tensor for name, tensor in self.frl_w.items() if 'alpha' in name]
                # # self.alpha_weights.append(self.pos_emb)
                # self.alpha_grads = tf.gradients(self.alpha_q, self.alpha_weights, self.dloss_dQa_input)
                # self.train_alpha = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                #     zip(self.alpha_grads, self.alpha_weights))
                self.dloss_dQ_input = []
                self.agent_grads = []
                self.train_agent = []
                for i in range(self.num_agents):
                    #self.dloss_dQ_input.append(tf.placeholder(tf.float32, [None, self.num_actions], 'dloss_dQ%s_input' % i))
                    self.dloss_dQ_input.append(tf.placeholder(tf.float32, [None, self.agent_outlayer], 'dloss_dQ%s_input' % i))
                    self.agent_grads.append(tf.gradients(self.agent_q[i], self.agent_weights[i], self.dloss_dQ_input[i]))
                    self.train_agent.append(tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                        zip(self.agent_grads[i], self.agent_weights[i])))

        tf.global_variables_initializer().run()

    def update_q_network_op(self, t_w, name):
        # self.alpha_t_w_input, self.alpha_t_w_assign_op = self.update_q_network_op(self.alpha_t_w,
        #                                                                           'alpha_update_q_network_op')
        with tf.variable_scope(name):
            t_w_input = {}
            t_w_assign_op = {}

            for name in t_w:
                t_w_input[name] = tf.placeholder(tf.float32, t_w[name].get_shape().as_list(), name)
                t_w_assign_op[name] = t_w[name].assign(t_w_input[name])

            return t_w_input, t_w_assign_op

    def update_target_network(self):
        if self.args.train_mode == 'single_alpha':
            for name in self.alpha_w:
                self.alpha_t_w_assign_op[name].eval({self.alpha_t_w_input[name]: self.alpha_w[name].eval()})

        elif self.args.train_mode == 'single_beta':
            for name in self.beta_w:
                self.beta_t_w_assign_op[name].eval({self.beta_t_w_input[name]: self.beta_w[name].eval()})

        elif self.args.train_mode == 'full':
            for name in self.full_w:
                self.full_t_w_assign_op[name].eval({self.full_t_w_input[name]: self.full_w[name].eval()})

        else:
            if self.preset_lambda:
                # for name in self.alpha_w:
                #     self.alpha_t_w_assign_op[name].eval({self.alpha_t_w_input[name]: self.alpha_w[name].eval()})
                # for name in self.beta_w:
                #     self.beta_t_w_assign_op[name].eval({self.beta_t_w_input[name]: self.beta_w[name].eval()})

                for i in range(self.num_agents):
                    for name in self.agent_w[i]:
                        self.agent_t_w_assign_op[i][name].eval({self.agent_t_w_input[i][name]: self.agent_w[i][name].eval()})

            else:
                for name in self.frl_w:
                    #self.frl_t_w_assign_op[name].eval({self.frl_t_w_input[name]: self.frl_t_w[name].eval()})
                    self.frl_t_w_assign_op[name].eval({self.frl_t_w_input[name]: self.frl_w[name].eval()})

    def train(self, minibatch):
        # ipdb.set_trace()
        #pre_states_alpha, pre_states_beta, actions, rewards, post_states_alpha, post_states_beta, terminals = minibatch
        pre_states_all, actions, rewards, post_states_all, terminals = minibatch

        if self.args.train_mode == 'single_alpha':
            postq = self.alpha_t_q.eval({self.s_a: pre_states_all[0]})
            max_postq = np.max(postq, axis=1)
            targets = self.alpha_q.eval({self.s_a: pre_states_all[0]})

        elif self.args.train_mode == 'single_beta':
            postq = self.beta_t_q.eval({self.s_b: pre_states_all[1]})
            max_postq = np.max(postq, axis=1)
            targets = self.beta_q.eval({self.s_b: pre_states_all[1]})

        elif self.args.train_mode == 'full':
            postq = self.full_t_q.eval({self.s_b: pre_states_all[1], self.s_a: pre_states_all[0]})
            max_postq = np.max(postq, axis=1)
            targets = self.full_q.eval({self.s_b: pre_states_all[1], self.s_a: pre_states_all[0]})

        else:  # train_mode is alpha, beta or frl
            # ipdb.set_trace()
            if self.preset_lambda:
                # beta_postq = self.beta_t_q.eval({self.s_b: post_states_all[1]})
                # alpha_postq = self.alpha_t_q.eval({self.s_a: post_states_all[0]})
                # max_postq = np.max(self.lambda_ * alpha_postq + (1 - self.lambda_) * beta_postq, axis=1)
                # beta_preq = self.beta_q.eval({self.s_b: pre_states_all[1]})
                # alpha_preq = self.alpha_q.eval({self.s_a: pre_states_all[0]})
                # targets = self.lambda_ * alpha_preq + (1 - self.lambda_) * beta_preq

                agent_postq = []
                for i in range(self.num_agents):
                    agent_postq1 = self.agent_t_q[i].eval({self.s_all[i]: post_states_all[i]})
                    agent_postq.append(agent_postq1)

                post_q_all = agent_postq[0] / self.num_agents
                for i in range(self.num_agents - 1):
                    post_q_all = post_q_all + agent_postq[i+1] / self.num_agents
                max_postq = np.max(post_q_all, axis=1)

                agent_preq = []
                for i in range(self.num_agents):
                    agent_preq1 = self.agent_q[i].eval({self.s_all[i]: pre_states_all[i]})
                    agent_preq.append(agent_preq1)

                targets = agent_preq[0] / self.num_agents
                for i in range(self.num_agents-1):
                    targets = targets + agent_preq[i+1] / self.num_agents

            else:
                # post_q_alpha = self.alpha_t_q.eval({self.s_a: post_states_all[0]}) #post_q_alpha: hist_len (32) , number_actions 25
                # post_q_beta = self.beta_t_q.eval({self.s_b: post_states_all[1]})
                # pre_q_alpha = self.alpha_q.eval({self.s_a: pre_states_all[0]})
                # pre_q_beta = self.beta_q.eval({self.s_b: pre_states_all[1]})
                post_q_agent, pre_q_agent = [], []
                for i in range(self.num_agents):
                    post_q_agent.append(self.agent_t_q[i].eval({self.s_all[i]: post_states_all[i]}))
                    pre_q_agent.append(self.agent_q[i].eval({self.s_all[i]: pre_states_all[i]}))

                if self.add_train_noise and np.random.rand() <= self.noise_prob:
                    # add Gaussian noise to Q-values with self.noise_prob probility
                    noise_alpha = np.random.normal(0.0, self.stddev, post_q_alpha.shape)
                    noise_beta = np.random.normal(0.0, self.stddev, post_q_beta.shape)
                    post_q_alpha += noise_alpha
                    post_q_beta += noise_beta
                    noise_alpha = np.random.normal(0.0, self.stddev, pre_q_alpha.shape)
                    noise_beta = np.random.normal(0.0, self.stddev, pre_q_beta.shape)
                    pre_q_alpha += noise_alpha
                    pre_q_beta += noise_beta

                # postq = self.frl_t_q.eval({self.alpha_q_input: post_q_alpha, self.beta_q_input: post_q_beta}) #postq hist_len (32) , number_actions 25
                # max_postq = np.max(postq, axis=1) #max_postq hist_len (32)
                # targets = self.frl_q.eval({self.alpha_q_input: pre_q_alpha, self.beta_q_input: pre_q_beta})
                #TODO mutiagent
                postq = self.frl_t_q.eval({self.agent_q_input[0]: post_q_agent[0], #.eval(agent_q_input); " "
                                           self.agent_q_input[1]: post_q_agent[1]
                                           # self.agent_q_input[2]: post_q_agent[2],
                                           # self.agent_q_input[3]: post_q_agent[3],
                                           # self.agent_q_input[4]: post_q_agent[4]
                                           })  # postq hist_len (32) , number_actions 25
                max_postq = np.max(postq, axis=1)  # max_postq hist_len (32)
                targets = self.frl_q.eval({self.agent_q_input[0]: pre_q_agent[0],
                                           self.agent_q_input[1]: pre_q_agent[1]
                                           # self.agent_q_input[2]: pre_q_agent[2],
                                           # self.agent_q_input[3]: pre_q_agent[3],
                                           # self.agent_q_input[4]: pre_q_agent[4]
                                           })
        # update done actions' value
        # for i, action in enumerate(actions): #actions: hist_len's actions 32
        #     if terminals[i]:
        #         targets[i, action] = rewards[i]
        #     else:
        #         targets[i, action] = rewards[i] + self.gamma * max_postq[i] #gamma = 1

        for i, action in enumerate(actions): #actions: hist_len's actions 32
            if terminals[i]:
                targets[i, action] = rewards[i]
            else:
                targets[i, action] = rewards[i] + self.gamma * max_postq[i] #gamma = 1

        if self.args.train_mode == 'single_alpha':
            _, delta, loss = self.sess.run([self.train_single_alpha,
                                            self.delta_alpha,
                                            self.loss_alpha
                                            ],
                                           {self.s_a: pre_states_all[0],
                                            self.target_q: targets
                                            })

        elif self.args.train_mode == 'single_beta':
            _, delta, loss = self.sess.run([self.train_single_beta,
                                            self.delta_beta,
                                            self.loss_beta
                                            ],
                                           {self.s_b: pre_states_all[1],
                                            self.target_q: targets
                                            })

        elif self.args.train_mode == 'full':
            _, delta, loss = self.sess.run([self.train_full_dqn,
                                            self.delta_full,
                                            self.loss_full
                                            ],
                                           {self.s_b: pre_states_all[1],
                                            self.s_a: pre_states_all[0],
                                            self.target_q: targets
                                            })

        elif self.args.train_mode == 'frl_lambda':
            _, delta, loss = self.sess.run([self.train_frl,
                                            self.delta_frl,
                                            self.loss_frl
                                            ],
                                           {#self.s_b: pre_states_beta,
                                            #self.s_a: pre_states_alpha,
                                            # self.s_b: pre_states_all[1],
                                            # self.s_a: pre_states_all[0],
                                            self.s_all[0]: pre_states_all[0],
                                            self.s_all[1]: pre_states_all[1],
                                            #self.s_all[2]: pre_states_all[2],
                                            #self.s_all: pre_states_all,
                                            self.target_q: targets
                                            })

        elif self.args.train_mode == 'frl_separate':
            # update parameters of mlp netowrk
            # _, dloss_dQa, dloss_dQb, delta, loss = self.sess.run([self.train_mlp, #update mlp weights
            #                                                       self.dloss_dQa, #get gradient
            #                                                       self.dloss_dQb,
            #                                                       self.delta_frl, #self.delta_frl = self.target_q - self.frl_q
            #                                                       self.loss_frl
            #                                                       ],
            #                                                      {self.alpha_q_input: pre_q_alpha,
            #                                                       self.beta_q_input: pre_q_beta,
            #                                                       self.target_q: targets
            #                                                       })
            _, dloss_dQ, delta, loss = self.sess.run([self.train_mlp,  # update mlp weights
                                                                  self.dloss_dQ,  # get gradient
                                                                  self.delta_frl,
                                                                  # self.delta_frl = self.target_q - self.frl_q
                                                                  self.loss_frl
                                                                  ],
                                                                 {self.agent_q_input[0]: pre_q_agent[0],
                                                                  self.agent_q_input[1]: pre_q_agent[1],
                                                                  #self.agent_q_input[2]: pre_q_agent[2],
                                                                  # self.agent_q_input[3]: pre_q_agent[3],
                                                                  # self.agent_q_input[4]: pre_q_agent[4],
                                                                  self.target_q: targets
                                                                  }) #TODO mutiagent
            for i in range(self.num_agents):
                self.sess.run([self.train_agent[i]
                               ],
                              {self.dloss_dQ_input[i]: dloss_dQ[i][0],  # both 32,16
                               self.s_all[i]: pre_states_all[i]
                               # self.s_b: pre_states_beta
                               })
            # # update parameters of beta network
            # self.sess.run([self.train_beta
            #                ],
            #               {self.dloss_dQb_input: dloss_dQb[0], #both 32,16
            #                self.s_b: pre_states_all[1]
            #                #self.s_b: pre_states_beta
            #                })
            # # update parameters of alpha network
            # self.sess.run([self.train_alpha
            #                ],
            #               {self.dloss_dQa_input: dloss_dQa[0],
            #                self.s_a: pre_states_all[0]
            #                #self.s_a: pre_states_alpha
            #                })

        else:
            print('\n Wrong training mode! \n')
            raise ValueError

        return loss


    #def predict(self, state_alpha, state_beta, predict_net):
    def predict(self, states_all, predict_net):
        # ipdb.set_trace()
        # state_alpha = np.transpose(state_alpha, (0, 1))
        # state_beta = np.transpose(state_beta, (0, 1))
        states_all = np.transpose(states_all, (0, 1, 2))
        if predict_net == 'alpha':
            qvalue = self.alpha_q.eval({self.s_a: states_all[0]})

        elif predict_net == 'beta':
            qvalue = self.beta_q.eval({self.s_b: states_all[1]})

        elif predict_net == 'full':
            qvalue = self.full_q.eval({self.s_a: states_all[0], self.s_b: states_all[1]})

        elif predict_net == 'both':
            # q_beta = self.beta_q.eval({self.s_b: states_all[1]})
            # q_alpha = self.alpha_q.eval({self.s_a: states_all[0]})
            # q_beta = self.beta_q.eval({self.s_b: state_beta})
            # q_alpha = self.alpha_q.eval({self.s_a: state_alpha})
            q_all = []
            for i in range(self.num_agents):
                q_all1 = self.agent_q[i].eval({self.s_all[i]: states_all[i]})
                q_all.append(q_all1)
            if self.preset_lambda:
                #qvalue = self.lambda_ * q_alpha + (1 - self.lambda_) * q_beta
                qvalue = q_all[0] / self.num_agents
                for i in range(self.num_agents-1):
                    qvalue = qvalue + q_all[i+1] / self.num_agents

            else:
                if self.add_predict_noise:
                    noise_alpha = np.random.normal(0.0, self.stddev, q_alpha.shape)
                    noise_beta = np.random.normal(0.0, self.stddev, q_beta.shape)
                    q_alpha += noise_alpha
                    q_beta += noise_beta
                #qvalue = self.frl_q.eval({self.alpha_q_input: q_alpha, self.beta_q_input: q_beta})
                #TODO mutiagent
                qvalue = self.frl_q.eval(
                    {self.agent_q_input[0]: q_all[0], self.agent_q_input[1]: q_all[1]})
                     #self.agent_q_input[2]: q_all[2], self.agent_q_input[3]: q_all[3],
                     #self.agent_q_input[4]: q_all[4]})
        else:
            print('\n Wrong predict mode! \n')
            raise ValueError

        return qvalue[0]

    def save_weights(self, weight_dir, net_name):
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        if net_name == 'full':
            print('Saving full network weights ...')
            for name in self.full_w:
                save_pkl(self.full_w[name].eval(), os.path.join(weight_dir, "full_%s.pkl" % name))

        elif net_name == 'beta':
            print('Saving beta network weights ...')
            for name in self.beta_w:
                save_pkl(self.beta_w[name].eval(), os.path.join(weight_dir, "beta_%s.pkl" % name))

        elif net_name == 'alpha':
            print('Saving alpha network weights ...')
            for name in self.alpha_w:
                save_pkl(self.alpha_w[name].eval(), os.path.join(weight_dir, "alpha_%s.pkl" % name))

        else:
            if self.preset_lambda:
                print('Saving frl preset_lambda network weights ...')
                for name in self.beta_w:
                    save_pkl(self.beta_w[name].eval(), os.path.join(weight_dir, "beta_%s.pkl" % name))

                for name in self.alpha_w:
                    save_pkl(self.alpha_w[name].eval(), os.path.join(weight_dir, "alpha_%s.pkl" % name))
            else:
                print('Saving frl mlp network weights ...')
                for name in self.frl_w:
                    save_pkl(self.frl_w[name].eval(), os.path.join(weight_dir, "frl_%s.pkl" % name))

    def load_weights(self, weight_dir):
        print('Loading weights from %s ...' % weight_dir)
        if self.args.train_mode == 'full':
            self.full_w_input, self.full_w_assign_op = self.update_q_network_op(self.full_w, 'load_full_pred_from_pkl')
            for name in self.full_w:
                self.full_w_assign_op[name].eval(
                    {self.full_w_input[name]: load_pkl(os.path.join(weight_dir, "full_%s.pkl" % name))})

        elif self.args.train_mode == 'frl_separate':
            self.frl_t_w_input, self.frl_w_assign_op = self.update_q_network_op(self.frl_w, 'load_frl_pred_from_pkl')
            for name in self.frl_w:
                self.frl_w_assign_op[name].eval(
                    {self.frl_t_w_input[name]: load_pkl(os.path.join(weight_dir, 'frl_%s.pkl' % name))})

        else:
            self.beta_w_input, self.beta_w_assign_op = self.update_q_network_op(self.beta_w, 'load_beta_pred_from_pkl')
            for name in self.beta_w:
                self.beta_w_assign_op[name].eval(
                    {self.beta_w_input[name]: load_pkl(os.path.join(weight_dir, "beta_%s.pkl" % name))})

            self.alpha_w_input, self.alpha_w_assign_op = self.update_q_network_op(self.alpha_w,
                                                                                  'load_alpha_pred_from_pkl')
            for name in self.alpha_w:
                self.alpha_w_assign_op[name].eval(
                    {self.alpha_w_input[name]: load_pkl(os.path.join(weight_dir, "alpha_%s.pkl" % name))})

        self.update_target_network()
