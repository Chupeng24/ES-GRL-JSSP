import numpy as np
from Params import configs
from pyjssp.simulators import JSSPSimulator
import time
from os import path

np.random.seed(configs.np_seed_train)

d = path.dirname(__file__)
parent_path = path.dirname(d) #获得d所在的目录,即d的父级目录


if __name__ == '__main__':
    env = JSSPSimulator(num_jobs=None, num_machines=None)
    env.reset(jssp_path= parent_path + '/benchmarks/FT10.txt',sched_ratio=0.2)
    disrule_name = "FIFO"
    start_time = time.time()
    while True:
        if env.random_stop_flag is True:
            fea, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name)
        else:
            fea, _, _, _, _,_, done = env.step(action=None,disrule_name=None)
        if done:
            print("makespan: ",env.global_time)
            end_time = time.time()
            interval = end_time - start_time
            print("calculate time:",interval)
            break

    env = JSSPSimulator(num_jobs=None, num_machines=None)
    env.reset(jssp_path= parent_path + '/benchmarks/FT10.txt',sched_ratio=0.2)
    disrule_name = "FIFO"
    start_time = time.time()
    while True:
        if env.random_stop_flag is True:
            fea, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name)
        else:
            fea, _, _, _, _,_, done = env.step(action=None,disrule_name=None)
        if done:
            print("makespan: ",env.global_time)
            end_time = time.time()
            interval = end_time - start_time
            print("calculate time:",interval)
            break