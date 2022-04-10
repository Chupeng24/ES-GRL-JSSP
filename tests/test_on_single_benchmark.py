import random

import numpy as np
from Params import configs
from pyjssp.simulators import JSSPSimulator
import time
import os
from pyjssp.configs import benchmarks_name_dict
import pandas as pd

np.random.seed(configs.np_seed_validation)
random.seed(configs.python_seed)

d = os.path.dirname(__file__)
parent_path = os.path.dirname(d) #获得d所在的目录,即d的父级目录

def test_on_single_instance(benchname,disrule_name=None,proctime_std=0,sched_ratio=None,mbrk_Ag=0.0,mbrk_seed=None):
    env = JSSPSimulator(num_jobs=None, num_machines=None)
    env.reset(jssp_path=parent_path + f'/benchmark/{benchname}.txt',
              proctime_std=proctime_std,sched_ratio=sched_ratio,mbrk_Ag=mbrk_Ag,mbrk_seed=mbrk_seed)
    # env.reset(jssp_path='FT10.txt',proctime_std=0)
    #start_time = time.time()
    while True:
        if env.random_stop_flag == True:
            _, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name)
        else:
            _, _, _, _, _,_, done = env.step(action=None,disrule_name=None)
        if done:
            #print("makespan: ",env.global_time)
            #end_time = time.time()
            #interval = end_time - start_time
            #print("calculate time:",interval)
            #env.draw_gantt_chart("rule_test_result.html", f'benchmark:{benchname}',10000)
            env.draw_gantt_chart(path=f"validate mbdatime{disrule_name}.html",benchmark_name= benchname,max_x = 100)
            print("env.global_time:",env.global_time)
            sum_mbda_time =  0
            if env.mbrk_Ag is not None and env.mbrk_Ag > 0:
                for _, machine in env.machine_manager.machines.items():
                    sum_mbda_time += machine.mbdatime
            print("sum_mbda_time:",sum_mbda_time)
            return env.global_time

            # break


if __name__ == '__main__':

    # experiment_2: test on all benchmark with stochastic processing time
    np.random.seed(configs.np_seed_validation)
    random.seed(configs.python_seed)
    output_flag_2 = True
    # dispatching_rules_list = [None, "FIFO", "LIFO", "SPT", "LPT", "STPT", "LTPT", "LOR", "MOR", "MWKR", "FDD/MWKR"]
    # proctime_std = [0, 1, 2, 3]
    dispatching_rules_list = ["FIFO", ]
    proctime_std = [0]
    benchmark_list = ["FT/ft06"]
    result_dict = {}
    op_list = ["mean","max","min"]
    index_list = []
    for benchmark in benchmark_list:
        for std in proctime_std:
            for rule in dispatching_rules_list:
                print("benchmark_name:",benchmark,",time std:",std,",dispatching rule:",rule)
                makespans = []
                for idx in range(1):
                    # makespans.append(test_on_single_instance(benchname=benchmark,
                    #                                          disrule_name=rule,proctime_std=std,sched_ratio=None))
                    makespans.append(test_on_single_instance(benchname=benchmark,
                                                             disrule_name=rule,mbrk_Ag=0.5, mbrk_seed = 9))
                print(makespans[0])
                print("####################################################################################################")
                if rule not in result_dict.keys():
                    result_dict[rule] = []
                mean_mp, max_mp, min_mp = np.mean(makespans), np.max(makespans), np.min(makespans)
                result_dict[rule].append(mean_mp)
                result_dict[rule].append(max_mp)
                result_dict[rule].append(min_mp)
                print(np.mean(makespans),'\n',np.max(makespans),'\n',np.min(makespans))
            for val in op_list:
                index_list.append(f'{benchmark}-{std}-{val}')

    # if output_flag_2:
    #     makespan_data_pd = pd.DataFrame(result_dict, index=index_list)
    #     writer = pd.ExcelWriter(f'experiment_2 result {TIMESTAMP}.xlsx')
    #     makespan_data_pd.to_excel(writer,float_format='%.3f')
    #     writer.save()
    #     print("experiment_2 result output success")