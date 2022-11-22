import random

from uniform_instance_gen import uni_instance_gen
import numpy as np
from Params import configs
from pyjssp.simulators import JSSPSimulator
from pyjssp.utils import Timer
import time

from pyjssp.configs import benchmarks_name_dict
#from os import path
import profile

np.random.seed(200)
random.seed(200)
seed_array = np.random.randint(low=0,high=1000000,size=1000000)
global_idx = 0

data_generator = uni_instance_gen


def test_on_single_instance(machine_matrix, processing_time_matrix,disrule_name=None):

    env = JSSPSimulator(num_jobs=None, num_machines=None)
    start_time = time.time()
    # env.reset(machine_matrix=machine_matrix, processing_time_matrix=processing_time_matrix)
    env.reset(machine_matrix=machine_matrix,
              processing_time_matrix=processing_time_matrix)
    while True:
        if env.random_stop_flag == True:
            _, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name,observe=True)
        else:
            _, _, _, _, _,_, done = env.step(action=None,disrule_name=None)
        if done:
            end_time = time.time()
            interval = end_time - start_time
            return env.global_time, interval
            # print("calculate time:",interval)
            #env.draw_gantt_chart("SPT_rule_test_result.html", "Ft06",100)


if __name__ == '__main__':


    dispatching_rules_list = ["FIFO","MOR","FDD/MWKR"]

    for rule in dispatching_rules_list:
        print("#################","using dispatching_rules:",rule,"#################")
        # tnum_m = 5
        # list_tnum_n = [5, 10, 15, 20, 25, 30]
        global_idx = 0
        tnum_m = 10
        list_tnum_n = [10, 20, 30, 40, 50, 60]
        for tnum_n in list_tnum_n:
            com_time = []
            for i in range(20):
                np.random.seed(seed_array[global_idx])
                proctime_matrix, m_matrix = data_generator(n_j=tnum_n, n_m=tnum_m, low=1, high=100)
                global_idx = global_idx + 1
                makespan,interval = test_on_single_instance(machine_matrix=m_matrix,
                                                            processing_time_matrix=proctime_matrix,
                                                            disrule_name=rule)
                com_time.append(interval)
            print(np.mean(np.array(com_time)))

        # tnum_n = 30
        # list_tnum_m = [5, 10, 15, 20, 25, 30]
        tnum_n = 60
        list_tnum_m = [10, 20, 30, 40, 50, 60]
        for tnum_m in list_tnum_m:
            com_time = []
            for i in range(20):
                np.random.seed(seed_array[global_idx])
                proctime_matrix, m_matrix = data_generator(n_j=tnum_n, n_m=tnum_m, low=1, high=100)
                global_idx = global_idx + 1
                makespan,interval = test_on_single_instance(machine_matrix=m_matrix,
                                                            processing_time_matrix=proctime_matrix,
                                                            disrule_name=rule)
                com_time.append(interval)
            print(np.mean(np.array(com_time)))
