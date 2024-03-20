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

data_generator = uni_instance_gen


def test_on_single_instance(machine_matrix, processing_time_matrix,disrule_name=None):

    env = JSSPSimulator(num_jobs=None, num_machines=None)
    # env.reset(machine_matrix=machine_matrix, processing_time_matrix=processing_time_matrix)
    env.reset(machine_matrix=m_matrix, processing_time_matrix=proctime_matrix)
    while True:
        if env.random_stop_flag == True:
            _, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name)
        else:
            _, _, _, _, _,_, done = env.step(action=None,disrule_name=None)
        if done:
            return env.global_time
            # print("calculate time:",interval)
            #env.draw_gantt_chart("SPT_rule_test_result.html", "Ft06",100)


if __name__ == '__main__':
    vali_data3 = []
    np.random.seed(200)
    random.seed(200)
    for i in range(100):
        n_m = 6
        n_j = 6
        proctime_matrix,m_matrix= data_generator(n_j=n_j, n_m=n_m, low=configs.low, high=configs.high)
        vali_data3.append((proctime_matrix,m_matrix))

    env = JSSPSimulator(num_jobs=None, num_machines=None)
    dispatching_rules_list = ["LIFO"]
    # dispatching_rules_list = [None,"FIFO", "LIFO", "SPT", "LPT", "STPT", "LTPT", "LOR", "MOR", "MWKR", "FDD/MWKR"]
    # timer = Timer("spend time of scheduling all instance by one dispatching rule ")
    # dispatching_rules_list = [None]
    # with timer:
    for rule in dispatching_rules_list:
        print("using dispatching_rules:",rule)
        makespan_list = []
        for data in vali_data3:
            proctime_matrix = data[0]
            m_matrix = data[1]
            makespan = test_on_single_instance(machine_matrix=m_matrix,
                                               processing_time_matrix=proctime_matrix,
                                               disrule_name=rule)
            makespan_list.append(makespan)
        for item in makespan_list:
            print(item)
        print("validation quality | mekespan mean:",np.mean(makespan_list))






