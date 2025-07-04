#coding:utf-8
import time
import ipdb
import argparse
import tensorflow as tf
import scipy.io

from replay_memory_muti import ReplayMemory
# from environment_muti import Environment
# from deep_q_network_muti import FRLDQN
# from agent_muti import Agent

from deep_q_network_branch import FRLDQN
from agent_branch import Agent
from environment_branch import Environment


# from deep_q_network_branch17 import FRLDQN
# from deep_q_network_branch15 import FRLDQN
from deep_q_network_branch55 import FRLDQN

# from agent_test import Agent
# from environment_test import Environment
# from replay_memory_test import ReplayMemory
# from deep_q_network_test import FRLDQN

#from deep_q_network_froze import FRLDQN

# from agent import Agent
# from deep_q_network import FRLDQN
# from environment import Environment
# from replay_memory import ReplayMemory
from utils import get_time, str2bool



def args_init():
    parser = argparse.ArgumentParser()
    numofagents = 55
    envarg = parser.add_argument_group('Environment')
    envarg.add_argument("--image_dim",              type=int,       default=8,    help="")
    envarg.add_argument("--state_dim",              type=int,       default=3,     help="")
    #envarg.add_argument("--hist_len",               type=int,       default=2,    help="")
    envarg.add_argument("--hist_len", type=int, default=1, help="")
    envarg.add_argument("--max_steps",              type=int,       default=2,     help="")
    envarg.add_argument("--image_padding",          type=int,       default=1,     help="")
    envarg.add_argument("--max_train_doms",         type=int,       default=6400,  help="")
    envarg.add_argument("--start_valid_dom",        type=int,       default=6400,  help="")
    envarg.add_argument("--start_test_dom",         type=int,       default=7200,  help="")
    envarg.add_argument("--automax",                type=int,       default=2,     help="")
    envarg.add_argument("--autolen",                type=int,       default=1,     help="")
    envarg.add_argument("--use_instant_distance",   type=int,       default=1,      help="")
    envarg.add_argument("--step_reward",            type=float,     default=-1.0,   help="")
    envarg.add_argument("--collision_reward",       type=float,     default=-10.0,  help="")
    envarg.add_argument("--terminal_reward",        type=float,     default=+50.0,  help="")
    envarg.add_argument("--num_agents", type=int, default=numofagents, help="")

    memarg = parser.add_argument_group('Replay memory')
    memarg.add_argument("--positive_rate",      type=float, default=0.9,    help="")
    memarg.add_argument("--reward_bound",       type=float, default=0.0,    help="")
    memarg.add_argument("--priority",           type=int,   default=1,      help="")
    memarg.add_argument("--replay_size",        type=int,   default=100000, help="")

    netarg = parser.add_argument_group('Deep Q-learning network')
    netarg.add_argument("--autofilter",         type=int,       default=1,      help="")
    #netarg.add_argument("--batch_size",         type=int,       default=32,     help="")
    netarg.add_argument("--batch_size", type=int, default=24*10, help="")
    #netarg.add_argument("--num_actions",        type=int,       default=16,     help="")
    #netarg.add_argument("--num_actions", type=int, default=5**numofagents, help="")
    netarg.add_argument("--num_action", type=int, default=5, help="")

    netarg.add_argument("--learning_rate",      type=float,     default=0.001,  help="") #original: 0.001
    netarg.add_argument("--gamma",              type=float,     default=0.995,    help="") #original: 0.9
    netarg.add_argument("--lambda_",            type=float,     default=0.5,    help="")

    #netarg.add_argument("--preset_lambda",      type=str2bool,  default=False,  help="")
    #netarg.add_argument("--preset_lambda", type=str2bool, default=True, help="")

    # netarg.add_argument("--add_train_noise",    type=str2bool,  default=True,   help="")
    # netarg.add_argument("--add_predict_noise",  type=str2bool,  default=True,   help="")
    netarg.add_argument("--add_train_noise", type=str2bool, default=False, help="")
    netarg.add_argument("--add_predict_noise", type=str2bool, default=False, help="")
    netarg.add_argument("--noise_prob",         type=float,     default=0.5,    help="")
    netarg.add_argument("--stddev",             type=float,     default=1.0,    help="")

    antarg = parser.add_argument_group('Agent')
    antarg.add_argument("--exploration_rate_start",     type=float, default=1,      help="")
    antarg.add_argument("--exploration_rate_end",       type=float, default=0.1,    help="")
    antarg.add_argument("--exploration_rate_test",      type=float, default=0.0,    help="")
    antarg.add_argument("--exploration_decay_steps",    type=int,   default=1000,   help="") #original:1000
    antarg.add_argument("--train_frequency",            type=int,   default=1,      help="")
    antarg.add_argument("--target_steps",               type=int,   default=5,      help="") #original:5
    antarg.add_argument("--save_result", type=int, default=0, help="")
    
    mainarg = parser.add_argument_group('Main loop')
    mainarg.add_argument("--gpu_fraction",      type=float,     default=0.2,        help="")
    mainarg.add_argument("--epochs",            type=int,       default=250,        help="")
    mainarg.add_argument("--start_epoch",       type=int,       default=0,          help="")
    mainarg.add_argument("--stop_epoch_gap",    type=int,       default=250,         help="")
    mainarg.add_argument("--success_base",      type=int,       default=-1,         help="")

    mainarg.add_argument("--load_weights",      type=str2bool,  default=False,      help="") ########################################
    #mainarg.add_argument("--load_weights", type=str2bool, default=True, help="")
    #mainarg.add_argument("--save_weights", type=str2bool, default=False, help="")
    mainarg.add_argument("--save_weights",      type=str2bool,  default=True,       help="")

    mainarg.add_argument("--predict_net",       type=str,       default='both',     help="")
    mainarg.add_argument("--result_dir",        type=str,       default="preset_lambda",     help="") #

    mainarg.add_argument("--train_mode",        type=str,       default='frl_separate',      help='') ########################################
    netarg.add_argument("--preset_lambda", type=str2bool, default=False, help="")
    # mainarg.add_argument("--train_mode",        type=str,       default='frl_lambda',     help='') #original
    # netarg.add_argument("--preset_lambda", type=str2bool, default=True, help="")

    mainarg.add_argument("--train_episodes",    type=int,       default=30,        help="") #original 100 #90
    mainarg.add_argument("--valid_episodes",    type=int,       default=10,        help="") #original 800
    mainarg.add_argument("--test_episodes",     type=int,       default=10,        help="") #original 800

    mainarg.add_argument("--test_multi_nets",   type=str2bool,  default=False,      help="") #
    
    args = parser.parse_args()
    if args.load_weights:
        args.exploration_rate_start = args.exploration_rate_end
    # if args.autolen:
    #     lens = {8: 2, 16: 4, 32 :8, 64: 16}
    #     args.hist_len = lens[args.image_dim]
    if not args.use_instant_distance:
        args.reward_bound = args.step_reward    # no collisions
    args.result_dir = 'results/{}_{}_im{}_s{}_his{}_{}.txt'.format(
        args.train_mode, args.predict_net, args.image_dim, args.state_dim, args.hist_len, args.result_dir)
    return args


def train_single_net(args):
    start = time.time()
    #time.process_time_ns()
    print('Current time is: %s' % get_time())
    print('Starting at train_multi_nets...')

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # Initial environment, replay memory, deep_q_net and agent
        #ipdb.set_trace()
        env = Environment(args)
        mem = ReplayMemory(args)
        net = FRLDQN(sess, args)
        agent = Agent(env, mem, net, args)

        best_result = {'valid': {'success_rate': {1: 0., 3: 0., 5: 0., 10: 0., -1 : 0.}, 'avg_reward': 0., 'log_epoch': -1, 'step_diff': -1},
                        'test': {'success_rate': {1: 0., 3: 0., 5: 0., 10: 0., -1 : 0.}, 'avg_reward': 0., 'log_epoch': -1, 'step_diff': -1}
        }

        # loop over epochs
        with open(args.result_dir, 'w') as outfile:
            print('\n Arguments:')
            outfile.write('\n Arguments:\n')
            for k, v in sorted(args.__dict__.items(), key=lambda x:x[0]):
                print('{}: {}'.format(k, v))
                outfile.write('{}: {}\n'.format(k, v))
            print('\n')
            outfile.write('\n')

            if args.load_weights:
                filename = 'weights/%s_%s.h5' % (args.train_mode, args.predict_net)
                net.load_weights(filename)

            try:
                for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
                    agent.train(epoch, args.train_episodes, outfile, args.predict_net)
                    rate, reward, diff, soc_a_record, soc_b_record = agent.test(epoch, args.test_episodes, outfile, args.predict_net, 'valid')

                    #if rate[args.success_base] > best_result['valid']['success_rate'][args.success_base]:
                    if reward > best_result['valid']['avg_reward']:
                        update_best(best_result, 'valid', epoch, rate, reward, diff)
                        print('best_epoch: {}\t best_success: {}\t avg_reward: {}\t step_diff: {}\n'.format(epoch, rate, reward, diff))
                        outfile.write('best_epoch: {}\t best_success: {}\t avg_reward: {}\t step_diff: {}\n\n'.format(epoch, rate, reward, diff))

                        rate, reward, diff, soc_a_record, soc_b_record = agent.test(epoch, args.test_episodes, outfile, args.predict_net, 'test')
                        update_best(best_result, 'test', epoch, rate, reward, diff)
                        print('\n Test results:\n success_rate: {}\t avg_reward: {}\t step_diff: {}\n'.format(rate, reward, diff))
                        outfile.write('\n Test results:\n success_rate: {}\t avg_reward: {}\t step_diff: {}\n'.format(rate, reward, diff))
                        soc_a_record_best = soc_a_record
                        soc_b_record_best = soc_b_record
                        if args.save_weights:
                            filename = 'weights/%s_%s.h5' % (args.train_mode, args.predict_net)
                            net.save_weights(filename, args.predict_net)
                            print('Saved weights %s ...\n' % filename)

                    if epoch - best_result['valid']['log_epoch'] >= args.stop_epoch_gap:
                        print('-----Early stopping, no improvement after %d epochs-----\n' % args.stop_epoch_gap)
                        print(soc_a_record)
                        print(soc_b_record)
                        break

            except KeyboardInterrupt:
                print('\n Manually kill the program ... \n')

            print('\n\n Best results:')
            outfile.write('\n\n Best results:\n')
            for data_flag, results in best_result.items():
                print('\t{}'.format(data_flag))
                outfile.write('\t{}\n'.format(data_flag))
                for k, v in results.items():
                    print('\t\t{}: {}'.format(k, v))
                    outfile.write('\t\t{}: {}\n'.format(k, v))
            end = time.time()
            outfile.write('\nTotal time cost: %ds\n' % (end - start))
            

    print('Current time is: %s' % get_time())
    print('Total time cost: %ds\n' % (end - start))



def train_multi_nets(args):
    #start = time.time()
    start = time.process_time_ns()
    print('Current time is: %s' % get_time())
    print('Starting at train_multi_nets...')

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # Initial environment, replay memory, deep_q_net and agent
        #ipdb.set_trace()
        env = Environment(args)
        mem = ReplayMemory(args)
        net = FRLDQN(sess, args)
        agent = Agent(env, mem, net, args)

        best_result = {'valid': {#'alpha': {'success_rate': {1: 0., 3: 0., 5: 0., 10: 0., -1 : 0.}, 'avg_reward': 0., 'log_epoch': -1, 'step_diff': -1},
                                 'beta': {'success_rate': {1: 0., 3: 0., 5: 0., 10: 0., -1 : 0.}, 'avg_reward': -1000000., 'log_epoch': -1, 'step_diff': -1},
                                 'both': {'success_rate': {1: 0., 3: 0., 5: 0., 10: 0., -1 : 0.}, 'avg_reward': -1000000., 'log_epoch': -1, 'step_diff': -1}},

                        'test': {#'alpha': {'success_rate': {1: 0., 3: 0., 5: 0., 10: 0., -1 : 0.}, 'avg_reward': 0., 'log_epoch': -1, 'step_diff': -1},
                                 'beta': {'success_rate': {1: 0., 3: 0., 5: 0., 10: 0., -1 : 0.}, 'avg_reward': -1000000., 'log_epoch': -1, 'step_diff': -1},
                                 'both': {'success_rate': {1: 0., 3: 0., 5: 0., 10: 0., -1 : 0.}, 'avg_reward': -1000000., 'log_epoch': -1, 'step_diff': -1}}
        }

        # loop over epochs
        with open(args.result_dir, 'w') as outfile:
            print('\n Arguments:')
            # outfile.write('\n Arguments:\n')
            # for k, v in sorted(args.__dict__.items(), key=lambda x:x[0]):
            #     print('{}: {}'.format(k, v))
            #     outfile.write('{}: {}\n'.format(k, v))
            # print('\n')
            # outfile.write('\n')

            if args.load_weights:
                filename = 'weights/%s_both.h5' % args.train_mode
                net.load_weights(filename)

            try:
                best_epoch = 0
                for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
                    agent.train(epoch, args.train_episodes, outfile, 'both')
                    #rate, reward, diff, soc_a_record, soc_b_record = agent.test(epoch, args.test_episodes, outfile, 'both', 'valid')
                    reward, soc_a_record, soc_b_record = agent.test(epoch, args.test_episodes,
                                                                                'both', 'valid', 0)
                    print('best_epoch: {}\n'.format(best_epoch))
                    rate = 0
                    diff = 0
                    #if rate[args.success_base] > best_result['valid']['both']['success_rate'][args.success_base]:
                    if reward > best_result['valid']['both']['avg_reward']:
                    #if epoch >= 40 and reward > -5600:
                        update_best(best_result, 'valid', epoch, rate, reward, diff, 'both')
                        best_epoch = epoch
                        print('best_epoch: {}\t best_success: {}\t avg_reward: {}\t step_diff: {}\n'.format(epoch, rate, reward, diff))
                        outfile.write('[both] \t best_epoch: {}\t best_success: {}\t avg_reward: {}\t step_diff: {}\n\n'.format(epoch, rate, reward, diff))

                        #rate, reward, diff, soc_a_record, soc_b_record = agent.test(epoch, args.test_episodes, outfile, 'both', 'test')
                        reward, soc_a_record, soc_b_record = agent.test(epoch, args.test_episodes, 'both', 'test', 0)
                        update_best(best_result, 'test', epoch, rate, reward, diff, 'both')
                        print('\n Test results:\t success_rate: {}\t avg_reward: {}\t step_diff: {}\n'.format(rate, reward, diff))
                        outfile.write('\n Test results:\t success_rate: {}\t avg_reward: {}\t step_diff: {}\n\n'.format(rate, reward, diff))
                        soc_a_record_best = soc_a_record
                        soc_b_record_best = soc_b_record
                        reward_best = reward
                        # if args.test_multi_nets:
                        # #for net_name in ['alpha', 'beta']:
                        #     net_name = 'beta'
                        #     #rate, reward, diff = agent.test(epoch, args.test_episodes, outfile, net_name, 'valid')
                        #     reward = agent.test(epoch, args.test_episodes, outfile, net_name, 'valid')
                        #     update_best(best_result, 'valid', epoch, rate, reward, diff, net_name)
                        #     print('best_epoch: {}\t best_success: {}\t avg_reward: {}\t step_diff: {}\n'.format(epoch, rate, reward, diff))
                        #     outfile.write('[{}] \t best_epoch: {}\t best_success: {}\t avg_reward: {}\t step_diff: {}\n\n'.format(net_name, epoch, rate, reward, diff))
                        #
                        #     #rate, reward, diff = agent.test(epoch, args.test_episodes, outfile, net_name, 'test')
                        #     reward = agent.test(epoch, args.test_episodes, outfile, net_name, 'test')
                        #
                        #     update_best(best_result, 'test', epoch, rate, reward, diff, net_name)
                        #     print('\n Test results:\t success_rate: {}\t avg_reward: {}\t step_diff: {}\n'.format(rate, reward, diff))
                        #     outfile.write('\n Test results:\t success_rate: {}\t avg_reward: {}\t step_diff: {}\n\n'.format(rate, reward, diff))


                        if args.save_weights:
                            filename = 'weights/%s_both.h5' % args.train_mode
                            net.save_weights(filename, 'both')
                            print('Saved weights %s ...\n' % filename)

                        outfile.write('\n')
                    
                    if epoch - best_result['valid']['both']['log_epoch'] >= args.stop_epoch_gap:
                    #if epoch >= 40 and reward > -5600:
                        print('-----Early stopping, no improvement after %d epochs-----\n' % args.stop_epoch_gap)
                        print(soc_a_record_best)
                        print(soc_b_record_best)
                        print(reward_best)
                        break
                #save result
                result_record1 = agent.save_record()
                scipy.io.savemat('result_record.mat', {'result_record': result_record1})
                print('\n Saved reward and loss result ... \n')
            except KeyboardInterrupt:
                print('\n Manually kill the program ... \n')

            print('\n\n Best results:')
            outfile.write('\n\n Best results:\n')
            for data_flag, results in best_result.items():
                print('\t{}'.format(data_flag))
                outfile.write('\t{}\n'.format(data_flag))
                for net_name, result in results.items():
                    print('\t\t{}'.format(net_name))
                    outfile.write('\t\t{}\n'.format(net_name))
                    for k, v in result.items():
                        print('\t\t\t{}: {}'.format(k, v))
                        outfile.write('\t\t\t{}: {}\n'.format(k, v))
            #end = time.time()
            end = time.process_time_ns()
            outfile.write('\nTotal time cost: %ds\n' % ((end - start) / 1000000000))

    proc_time = (end - start) / 1000000000

    print('Current time is: %s' % get_time())
    print('Total time cost: %ds\n' % proc_time)
    return proc_time


def update_best(result, data_flag, epoch, rate, reward, diff, net_name=''):
    if net_name:
        result[data_flag][net_name]['success_rate'] = rate
        result[data_flag][net_name]['avg_reward'] = reward
        result[data_flag][net_name]['log_epoch'] = epoch
        result[data_flag][net_name]['step_diff'] = diff
    else:
        result[data_flag]['success_rate'] = rate
        result[data_flag]['avg_reward'] = reward
        result[data_flag]['log_epoch'] = epoch
        result[data_flag]['step_diff'] = diff



num_try = 1
proc_time_record = []
for trys in range(num_try):
    if __name__ == '__main__':
        args = args_init()
        if args.train_mode in ['full', 'single_alpha', 'single_beta']:
            train_single_net(args)
        # else:
        #     train_multi_nets(args)
        else:
            #num_try = 10
            proc_time_record = []
            #for trys in range(num_try):
            proc_time = train_multi_nets(args)
            proc_time_record.append(proc_time)

            print('process_time_record: %s' % proc_time_record)
            print('process_time_average: %s\n' % (sum(proc_time_record)/num_try))

