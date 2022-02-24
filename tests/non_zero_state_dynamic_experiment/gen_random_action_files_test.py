import numpy as np
from Params import configs
from pyjssp.simulators import JSSPSimulator
import time
from os import path

np.random.seed(configs.np_seed_train)

n_j = 10
n_m = 10
np_seed_validation = 200
sched_ratio = 0.4
d = path.dirname(__file__)
parent_path = path.dirname(d) #获得d所在的目录,即d的父级目录
disrule_name = "FDD/MWKR"

# if __name__ == '__main__':
#     dataLoaded = np.load(parent_path + '/DataGen/generatedData' + str(n_j) + '_' + str(n_m) + '_Seed' + str(np_seed_validation) + '.npy')
#     load_data_set = []
#     for i in range(dataLoaded.shape[0]):
#         load_data_set.append((dataLoaded[i][0], dataLoaded[i][1]))
#     random_action_set = []
#
#     action_set = np.load(parent_path + '/DataGen/RandomActionSet/generatedData'+ str(n_j) + '_' + str(n_m) + '_Seed_' +
#                          str(np_seed_validation) + 'SchedRatio' + str(sched_ratio) + '.npy')
#
#     env = JSSPSimulator(num_jobs=None, num_machines=None)
#     makespans_1 = []
#     makespans_2 = []
#
#     for index, data in enumerate(load_data_set):
#         env.reset(processing_time_matrix=data[0], machine_matrix=data[1],sched_ratio=0)
#         random_action_index = 0
#         while True:
#             if np.isnan(action_set[index][random_action_index]):
#                 _, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name)
#             else:
#                 _, _, _, _, _,_, done = env.step(action_set[index][random_action_index])
#                 random_action_index += 1
#             if done:
#                 makespans_1.append(env.global_time)
#                 break
#
#     for index, data in enumerate(load_data_set):
#         env.reset(processing_time_matrix=data[0], machine_matrix=data[1],sched_ratio=0)
#         random_action_index = 0
#         while True:
#             if np.isnan(action_set[index][random_action_index]):
#                 _, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name)
#             else:
#                 _, _, _, _, _,_, done = env.step(action_set[index][random_action_index])
#                 random_action_index += 1
#             if done:
#                 makespans_2.append(env.global_time)
#                 break
#     makespans_1 = np.array(makespans_1)
#     makespans_2 = np.array(makespans_2)
#     compare_result = makespans_1 - makespans_2
#     print(compare_result)

# 另一个测试
if __name__ == '__main__':
    dataLoaded = np.load(parent_path + '/DataGen/generatedData' + str(n_j) + '_' + str(n_m) + '_Seed' + str(np_seed_validation) + '.npy')
    load_data_set = []
    for i in range(dataLoaded.shape[0]):
        load_data_set.append((dataLoaded[i][0], dataLoaded[i][1]))
    random_action_set = []

    # action_set = np.load(parent_path + '/DataGen/RandomActionSet/generatedData'+ str(n_j) + '_' + str(n_m) + '_Seed_' +
    #                      str(np_seed_validation) + 'SchedRatio' + str(sched_ratio) + '.npy')

    env = JSSPSimulator(num_jobs=None, num_machines=None)
    makespans_1 = []
    makespans_2 = []

    for index, data in enumerate(load_data_set):
        env.reset(processing_time_matrix=data[0], machine_matrix=data[1],sched_ratio=0.2)
        random_action_index = 0
        while True:
            if env.random_stop_flag==True:
                _, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name)
            else:
                _, _, _, _, _,_, done = env.step(action=None,disrule_name=None)
            if done:
                makespans_1.append(env.global_time)
                break
    for index, data in enumerate(load_data_set):
        env.reset(processing_time_matrix=data[0], machine_matrix=data[1],sched_ratio=0.2)
        random_action_index = 0
        while True:
            if env.random_stop_flag==True:
                _, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name)
            else:
                _, _, _, _, _,_, done = env.step(action=None,disrule_name=None)
            if done:
                makespans_2.append(env.global_time)
                break
    makespans_1 = np.array(makespans_1)
    makespans_2 = np.array(makespans_2)
    compare_result = makespans_1 - makespans_2
    print(compare_result)        # if all equals zero,the program for non-zero-state is right.