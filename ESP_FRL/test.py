#import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
states = sio.loadmat('voltdata18.mat')
print(states)
a = states["voltdata18"]
print(a[0])


a = 3
print(a!=2)
a = np.zeros([2,5], dtype=np.float16)
# a[1,0] =2
# a[2,0] =4
b = [[15], [17]]
b[0][0] = 11
b[1][0] = 15
a = np.append(a, b, axis = 1)
print(a)
print(len(a[0]))
# scipy.io.savemat('test.mat', {'test': app_arr})
# np.save('data.npy', a) # save
#
# new_num_arr = np.load('data.npy') # load
# print(new_num_arr)



#
# a = np.random.rand()
# b = np.random.rand()
# print(a)
# print(b)
#
# a = [[1,2,3],[4,5,6],[7,8,9]]
# b = a[1][0]
# print(b)
#
# a = np.array([[0,2,3],[0,5,6],[0,8,9]],dtype=np.uint8)
# b = a[:, 0]
# c = a[2][1]
# print(a)
# print(c)
# print(type(c))
#
# soc = [0.2, 0.3, 0.3]
# move = [0.1, 0.2]
#
#
# for i in range(3):
#     while True:
#         action = np.random.randint(2)
#         new_soc = soc[i] - move[action]
#         if -0.001 <= new_soc <= 1.001:
#             break

# num_try = 10
# proc_time_record = []
# for trys in range(num_try):
#     proc_time = trys* 2.66643
#     proc_time_record.append(proc_time)
#
# print('process_time_record: %s' % proc_time_record)
# print('process_time_average: %s\n' % (sum(proc_time_record)/num_try))
# start = time.process_time_ns()
# for i in range(1000):
#     for j in range(100):
#         for k in range(1000):
#             a = i+j+k
# end = time.process_time_ns()
# b = (end - start)/ 1000000000

# print (start)
# print (end)
# print (b)
soc_max = [25, 24, 32, 24, 32]


# actions = [117, 1, 2, 3, 4]
# action_all_batch =[]
# for k in range (5):
#     action_all =[]
#     action = actions[k]
#     for i in range(3):
#         action, a1 = divmod(action, 5)
#         action_all.insert(0, a1)
#     action_all_batch.append(action_all)
# action_all_batch = np.transpose(action_all_batch, (1, 0))
# print(action_all)
# print(action_all_batch)
# print(action_all_batch[0])
# postq = np.zeros([2, 10, 5], dtype=np.float16)
#
# for i in range(2):
#     selected_a = tf.argmax(postq[i], axis=1)
#     selected_q = tf.reduce_sum(tf.one_hot(selected_a, 5) * postq[i], axis=1)  # (?,)
#     if i == 0:
#         mean_next_q_values = selected_q
#     else:
#         mean_next_q_values += selected_q
# mean_next_q_values /= 5
# target_q_values = [1 + 1 * mean_next_q_values] * 5
# a = 1

# a = tf.Variable([2.0, 3.0])
# # Create b based on the value of a
# b = tf.Variable([2.])
# init_op = tf.global_variables_initializer()
# c = a + b
# with tf.Session() as sess:
#     sess.run(init_op) #execute init_op
#     #print the random values that we sample
#     print (sess.run(c))
#
# def ae():
#     a = 1
#     c= [2, 3]
#     return [a + b for b in c]

# a = tf.constant([[1., 1.], [2., 2.]])
# b = tf.reduce_mean(a, 1)
# with tf.Session() as sess:
#     sess.run(init_op) #execute init_op
#     #print the random values that we sample
#     print (sess.run(b))

#
# num_agents = 2
# num_actions = 5
# # alpha_postq = [[1.1, 1.1, 3.2] , [1.1, 1.1, 3.2]]
# # beta_postq = [[3.1, 4.1, 5.1], [3.1, 4.1, 5.1]]
# # # a = 0.5 * alpha_postq
# # # print(a)
# # print(1 * alpha_postq + 1 * beta_postq)
# #
# # max_postq = np.max(1 * alpha_postq + 1 * beta_postq, axis=1)
# # print(max_postq)
# agent_q_input = []
# for i in range(num_agents):
#     #self.agent_q_input.append(tf.placeholder(tf.float32, [None, self.state_dim], 's_%s' % i))
#     agent_q_input.append(tf.placeholder(tf.float32, [None, num_actions], '%s_q_input' % i))
#
# print(agent_q_input)
#
# alpha_q_input = tf.placeholder(tf.float32, [None, num_actions], 'alpha_q_input')
# beta_q_input = tf.placeholder(tf.float32, [None, num_actions], 'beta_q_input')
# print(alpha_q_input)
# print(beta_q_input)
#
# concat_q = tf.concat(agent_q_input, axis=1)
# print(concat_q)
#
# concat_q1 = tf.concat([alpha_q_input, beta_q_input], axis=1)
# print(concat_q1)



# def test1(a, b):
#     return a + b
#
# arr = [1, 2 ]
# c = test1(arr[0], arr[1])
# #c = (tensor for name, tensor in arr)
# for item in arr:
#     c = test1(item)
# #[tensor for name, tensor in self.frl_w.items() if 'alpha' in name]
#
# print(c)


#c = test1(arr[0:])

# for i in range(2):
#     aa =

#print(c)
# s_ = []
# i =1
# #s_1 = tf.placeholder(tf.float32, [None, 5], 's_%s' % i)
# s_.append(tf.placeholder(tf.float32, [None, 5], 's_%s' % i))
#
# s_2 = tf.placeholder(tf.float32, [None, 5], 's_b')
# s_.append(s_2)
#
# print(s_)

# agent_w = []
# beta_w = {1}
# agent_w.append(beta_w)
# beta_w = {2}
# agent_w.append(beta_w)
#
# print(agent_w)

# def getState():
#     states = np.zeros([2,1,1], dtype=np.int8)
#     return states
#
#
# states = getState()
# print(states)
# #qvalue = predict(states, predict_net)
#
# act = np.zeros([2], dtype=np.float16)
# action = 1
# for i in range(2):
#     action, act[-(i+1)] = divmod(action, 5) # 5^ N actions
# print(act)

# print(act_c)

# states_alpha = np.zeros([4, 1, 4], dtype=np.float32)
# print(states_alpha[1][0])

# data_type = 2
# dom = 7
#
# if data_type == 1:
#     dom += 1
#     if ((dom + 1) % 4) == 0:
#         dom += 1
# else:
#     dom += 4
# print(dom)


# for i in range(1, 1 + 1):
#     print(i)
#
# step = 1
# hour = step % 24
# print(hour)
#
#
#
# def is_valid_soc(soc):
#     return not (soc > 1 or soc < 0)
#
# print(is_valid_soc(-1))
#
# states_alpha = np.zeros([1, 4], dtype=np.float32)
# states_alpha[0] = [1, 2, 3, 0]
# print(states_alpha[0])


# states = sio.loadmat('pvloaddata.mat')
# states = states["pvloaddata"]
# print(states[0,1])
#
# states_alpha = np.zeros([1, 4, 3], dtype=np.float32)
# print(states_alpha[0,-1])
#
# array1 = [1, 2, 3 ,4 ,5]
# array1[ : -1] = array1[ 1:]
# print(array1)
#
# #self.states_alpha = np.zeros([1, self.hist_len, self.state_alpha_dim, self.state_alpha_dim], dtype=np.float32)
# states_alpha = np.zeros([1, 4, 3, 3], dtype=np.float32)
# # print(states_alpha)
# # print(states_alpha[0, : -1])
#
# reward = sio.loadmat('reward.mat')
# reward = reward["reward"]
# steps = 1000
# action = 24
# act_a, act_b = divmod(action, 5)
# move = [0, -0.025, -0.05, 0.025, 0.05]
# print(act_a)
# print(act_b)
# print(move[act_a])
# print(move[act_b])
# print(reward[steps][action-1])
