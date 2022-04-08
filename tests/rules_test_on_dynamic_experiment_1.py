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

#@profile
def test_on_all_instance(disrule_name=None,proctime_std=0,sched_ratio=None):
    path_list = []
    name_list = []
    # print(benchmarks_name_dict[benchsize])
    for root, dirs, files in os.walk(parent_path + f'/benchmark', topdown=False):
        for name in files:
            # print(name.split(".",1)[0])
            name_list.append(name.split(".",1)[0])
            path_list.append(os.path.join(root, name))
    # print(name_list)
    makespan_list = []
    env = JSSPSimulator(num_jobs=None, num_machines=None)
    for elem in path_list:
        env.reset(jssp_path=elem,proctime_std=proctime_std,sched_ratio=sched_ratio)
        # env.reset(jssp_path='FT10.txt',proctime_std=0)
        # start_time = time.time()
        while True:
            if env.random_stop_flag == True:
                _, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name)
            else:
                _, _, _, _, _,_, done = env.step(action=None,disrule_name=None)
            if done:
                #print(env.global_time)
                #end_time = time.time()
                #interval = end_time - start_time
                # print("calculate time:",interval)
                #env.draw_gantt_chart("SPT_rule_test_result.html", "Ft06",100)
                print(elem,":",env.global_time)
                makespan_list.append(env.global_time)
                break
    return makespan_list
    #print("mean_makespan:",np.mean(makespan_list))

def test_on_single_instance(benchname,disrule_name=None,proctime_std=0,proc_seed=None,sched_ratio=None,sched_seed = None, mbrk_Ag=None,mbrk_seed=None):
    env = JSSPSimulator(num_jobs=None, num_machines=None)
    env.reset(jssp_path=parent_path + f'/benchmark/{benchname}.txt',
              proctime_std=proctime_std, proc_seed=proc_seed, sched_ratio=sched_ratio, sched_seed = sched_seed, mbrk_Ag=mbrk_Ag,
              mbrk_seed=mbrk_seed)
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
            return env.global_time

            # break

def test_on_same_cls_instance(bench_cls_name,disrule_name=None,proctime_std=0,sched_ratio=None):
    name_list = []
    file_root = None
    # print(parent_path + f'/benchmark/{bench_cls_name}')
    for root, dirs, files in os.walk(parent_path + f'/benchmark/{bench_cls_name}', topdown=False):
        file_root = root
        for name in files:
            name_list.append(name)
    # print(name_list)
    make_spans = {}
    env = JSSPSimulator(num_jobs=None, num_machines=None)
    for elem in name_list:
        env.reset(jssp_path=os.path.join(file_root, elem),proctime_std=proctime_std,sched_ratio=sched_ratio)
        # env.reset(jssp_path='FT10.txt',proctime_std=0)
        start_time = time.time()
        while True:
            if env.random_stop_flag == True:
                _, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name)
            else:
                _, _, _, _, _,_, done = env.step(action=None,disrule_name=None)
            if done:
                print(elem," makespan: ",env.global_time)
                end_time = time.time()
                interval = end_time - start_time
                make_spans[elem] =  env.global_time
                # print("calculate time:",interval)
                #env.draw_gantt_chart("SPT_rule_test_result.html", "Ft06",100)
                break

def test_on_same_size_instance(benchsize,disrule_name=None,proctime_std=0,sched_ratio=None):
    name_list = []
    # print(benchmarks_name_dict[benchsize])
    for root, dirs, files in os.walk(parent_path + f'/benchmark', topdown=False):
        for name in files:
            if name.split(".",1)[0] in benchmarks_name_dict[benchsize]:
                print(name)
                name_list.append(os.path.join(root, name))
    # print(name_list)
    makespan_list = []
    env = JSSPSimulator(num_jobs=None, num_machines=None)
    for elem in name_list:
        env.reset(jssp_path=elem,proctime_std=proctime_std,sched_ratio=sched_ratio)
        # env.reset(jssp_path='FT10.txt',proctime_std=0)
        start_time = time.time()
        while True:
            if env.random_stop_flag == True:
                _, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name)
            else:
                _, _, _, _, _,_, done = env.step(action=None,disrule_name=None)
            if done:
                print("makespan: ",elem,env.global_time)
                makespan_list.append(env.global_time)
                end_time = time.time()
                interval = end_time - start_time
                # print("calculate time:",interval)
                #env.draw_gantt_chart("SPT_rule_test_result.html", "Ft06",100)
                break
    print(makespan_list)
    print("mean_makespan:",np.mean(makespan_list))

if __name__ == '__main__':
    TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))

    # #experiment_1: test on all benchmark with no dynamic event
    # np.random.seed(configs.np_seed_validation)
    # random.seed(configs.python_seed)
    # output_flag_1 = True
    # dispatching_rules_list = [None, "FIFO", "LIFO", "SPT", "LPT", "STPT", "LTPT", "LOR", "MOR", "MWKR", "FDD/MWKR"]
    # name_list = []
    # for root, dirs, files in os.walk(parent_path + f'/benchmark', topdown=False):
    #     for name in files:
    #         # print(name.split(".",1)[0])
    #         name_list.append(name.split(".",1)[0])
    #
    # ms_on_rule_dict = {}
    # for rule in dispatching_rules_list:
    #     print("using dispatching rule:",rule)
    #     ms_on_rule = test_on_all_instance(disrule_name=rule)
    #     print(np.mean(ms_on_rule))
    #     ms_on_rule_dict[rule] = ms_on_rule
    #
    # if output_flag_1:
    #     makespan_data_pd = pd.DataFrame(ms_on_rule_dict, index=name_list)
    #     writer = pd.ExcelWriter(f'experiment_1 result {TIMESTAMP}.xlsx')
    #     makespan_data_pd.to_excel(writer,float_format='%.3f')
    #     writer.save()
    #     print("experiment_1 result output success")
    print("strat")
    # experiment_2: test on all benchmark with stochastic processing time
    # np.random.seed(configs.np_seed_validation)
    # random.seed(configs.python_seed)
    # output_flag_2 = True
    # dispatching_rules_list = [None, "FIFO", "LIFO", "SPT", "LPT", "STPT", "LTPT", "LOR", "MOR", "MWKR", "FDD/MWKR"]
    # proctime_std = [1, 2, 3]
    # benchmark_list = ["LA/la01", "FT/ft20","LA/la26","SWV/swv11"]
    # result_dict = {}
    # op_list = ["mean","max","min"]
    # index_list = []
    # for benchmark in benchmark_list:
    #     for std in proctime_std:
    #         for rule in dispatching_rules_list:
    #             print("benchmark_name:",benchmark,",time std:",std,",dispatching rule:",rule)
    #             makespans = []
    #             for idx in range(50):
    #                 makespans.append(test_on_single_instance(benchname=benchmark,
    #                                                          disrule_name=rule,proctime_std=std,proc_seed=idx+10))
    #                                                         # sched_ratio=0.4,mbrk_Ag=0.05,mbrk_seed=idx+20))
    #             print(makespans)
    #             if rule not in result_dict.keys():
    #                 result_dict[rule] = []
    #             mean_mp, max_mp, min_mp = np.mean(makespans), np.max(makespans), np.min(makespans)
    #             result_dict[rule].append(mean_mp)
    #             result_dict[rule].append(max_mp)
    #             result_dict[rule].append(min_mp)
    #             print(np.mean(makespans),'\n',np.max(makespans),'\n',np.min(makespans))
    #         for val in op_list:
    #             index_list.append(f'{benchmark}-{std}-{val}')
    #
    # if output_flag_2:
    #     makespan_data_pd = pd.DataFrame(result_dict, index=index_list)
    #     writer = pd.ExcelWriter(f'experiment with stochastic processing time result {TIMESTAMP}.xlsx')
    #     makespan_data_pd.to_excel(writer,float_format='%.3f')
    #     writer.save()
    #     print("Dynamic experiment with stochastic processing time result output success")
    #
    # # experiment_3: test on all benchmark with non-zero states
    # output_flag_3 = True
    # dispatching_rules_list = [None, "FIFO", "LIFO", "SPT", "LPT", "STPT", "LTPT", "LOR", "MOR", "MWKR", "FDD/MWKR"]
    # # dispatching_rules_list = ["FIFO"]
    # benchmark_list = ["LA/la01", "FT/ft20","LA/la26","SWV/swv11"]
    # sched_ratio = [0.2, 0.4, 0.6]
    # result_dict = {}
    # op_list = ["mean","max","min"]
    # index_list = []
    # detail_res_list = {}
    # for benchmark in benchmark_list:
    #     for ratio in sched_ratio:
    #         for rule in dispatching_rules_list:
    #             makespans = []
    #             print("benchmark_name:",benchmark,",schedule ratio:",ratio,",dispatching rule:",rule)
    #             for idx in range(50):
    #                 makespans.append(test_on_single_instance(benchname=benchmark, disrule_name=rule, sched_ratio=ratio, sched_seed=idx+10))
    #             print(makespans)
    #             detail_res_list[f'{benchmark}-{ratio}-{rule}'] = makespans
    #             if rule not in result_dict.keys():
    #                 result_dict[rule] = []
    #             median_mp, max_mp, min_mp = np.median(makespans), np.max(makespans), np.min(makespans)
    #             result_dict[rule].append(median_mp)
    #             result_dict[rule].append(max_mp)
    #             result_dict[rule].append(min_mp)
    #             print(median_mp,'\n',max_mp,'\n',min_mp)
    #         for val in op_list:
    #             index_list.append(f'{benchmark}-{ratio}-{val}')
    #
    # if output_flag_3:
    #     makespan_data_pd = pd.DataFrame(result_dict, index=index_list)
    #     writer = pd.ExcelWriter(f'experiment_3 result {TIMESTAMP}.xlsx')
    #     makespan_data_pd.to_excel(writer,float_format='%.3f')
    #     writer.save()
    #
    #     makespan_data_pd = pd.DataFrame(detail_res_list, index=list(range(1,51)))
    #     writer = pd.ExcelWriter(f'experiment with non-zero states detailed result {TIMESTAMP}.xlsx')
    #     makespan_data_pd.to_excel(writer,float_format='%.3f')
    #     writer.save()
    #     print("Dynamic experiment with non-zero states result output success")

    # experiment_4: test on all benchmark with machine breakdown
    output_flag_4 = True
    dispatching_rules_list = [None, "FIFO", "LIFO", "SPT", "LPT", "STPT", "LTPT", "LOR", "MOR", "MWKR", "FDD/MWKR"]
    Ag_list = [0.05] # [0.02, 0.05, 0.08]
    benchmark_list = ["LA/la26"] #["LA/la01", "FT/ft20", "LA/la26", "SWV/swv11"]
    result_dict = {}
    op_list = ["mean", "max", "min"]
    index_list = []
    for benchmark in benchmark_list:
        for Ag in Ag_list:
            for rule in dispatching_rules_list:
                print("benchmark_name:", benchmark, ",breakdown Ag:", Ag, ",dispatching rule:", rule)
                makespans = []
                for idx in range(50):
                    makespans.append(test_on_single_instance(benchname=benchmark,
                                                             disrule_name=rule, mbrk_Ag=Ag, mbrk_seed=idx + 10))
                print(makespans)
                if rule not in result_dict.keys():
                    result_dict[rule] = []
                mean_mp, max_mp, min_mp = np.mean(makespans), np.max(makespans), np.min(makespans)
                result_dict[rule].append(mean_mp)
                result_dict[rule].append(max_mp)
                result_dict[rule].append(min_mp)
                print(np.mean(makespans), '\n', np.max(makespans), '\n', np.min(makespans))
            for val in op_list:
                index_list.append(f'{benchmark}-{Ag}-{val}')

    if output_flag_4:
        makespan_data_pd = pd.DataFrame(result_dict, index=index_list)
        writer = pd.ExcelWriter(f'experiment with machine breakdown result {TIMESTAMP}.xlsx')
        makespan_data_pd.to_excel(writer, float_format='%.3f')
        writer.save()
        print("Dynamic experiment with machine breakdown output success")