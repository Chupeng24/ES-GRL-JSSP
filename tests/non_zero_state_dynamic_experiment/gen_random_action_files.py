import numpy as np
from Params import configs
from pyjssp.simulators import JSSPSimulator
import time
from os import path

np.random.seed(configs.np_seed_train)

n_j = 50
n_m = 20
np_seed_validation = 200
sched_ratio = 0.6
d = path.dirname(__file__)
parent_path = path.dirname(d) #获得d所在的目录,即d的父级目录
disrule_name = "FIFO"

if __name__ == '__main__':
    dataLoaded = np.load(parent_path + '/DataGen/generatedData' + str(n_j) + '_' + str(n_m) + '_Seed' + str(np_seed_validation) + '.npy')
    load_data_set = []
    for i in range(dataLoaded.shape[0]):
        load_data_set.append((dataLoaded[i][0], dataLoaded[i][1]))
    random_action_set = []

    env = JSSPSimulator(num_jobs=None, num_machines=None)

    for index, data in enumerate(load_data_set):
        env.reset(processing_time_matrix=data[0], machine_matrix=data[1],sched_ratio=sched_ratio)
        while True:
            if env.random_stop_flag is False:
                fea, _, _, _, _,_, done = env.step(action=None)
            else:
                random_action_set.append(env.random_action_list)
                break

    random_action_set = np.array(random_action_set)
    np.save(parent_path + '/DataGen/RandomActionSet/generatedData'+ str(n_j) + '_' + str(n_m) + '_Seed_' +
            str(np_seed_validation) + 'SchedRatio' + str(sched_ratio) + '.npy',random_action_set )