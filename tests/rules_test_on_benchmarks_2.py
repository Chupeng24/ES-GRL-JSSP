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

def test_on_single_instance(benchname,disrule_name=None,proctime_std=0,proc_seed=None,sched_ratio=None,mbrk_Ag=0.0,mbrk_seed=None):
    env = JSSPSimulator(num_jobs=None, num_machines=None)
    env.reset(jssp_path=parent_path + f'/benchmark/{benchname}.txt',
              proctime_std=proctime_std,proc_seed=proc_seed,sched_ratio=sched_ratio,mbrk_Ag=mbrk_Ag,mbrk_seed=mbrk_seed)
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

    #experiment_1: test on all benchmark with no dynamic event
    # np.random.seed(configs.np_seed_validation)
    # random.seed(configs.python_seed)
    # output_flag_1 = True
    # # dispatching_rules_list = [None, "FIFO", "LIFO", "SPT", "LPT", "STPT", "LTPT", "LOR", "MOR", "MWKR", "FDD/MWKR"]
    # dispatching_rules_list = [None, "FIFO", "LIFO", "SPT"]
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

    # experiment_2: test on all benchmark with stochastic processing time
    # from pyjssp.utils import Timer
    # Timer_exp_std = Timer("Dynamic experiment with proc time")
    # with Timer_exp_std:
    #     np.random.seed(configs.np_seed_validation)
    #     random.seed(configs.python_seed)
    #     output_flag_2 = True
    #     dispatching_rules_list = [None, "FIFO", "LIFO", "SPT", "LPT", "STPT", "LTPT", "LOR", "MOR", "MWKR", "FDD/MWKR"]
    #     # proctime_std = [0, 1, 2, 3]
    #     proctime_std = [1, 2, 3]
    #     # benchmark_list = ["FT/ft06", "LA/la16","DMU/dmu06","TA/ta61"]
    #     benchmark_list = ["LA/la16", "LA/la17", "LA/la18", "LA/la19", "LA/la20",
    #                       "DMU/dmu01", "DMU/dmu02", "DMU/dmu03", "DMU/dmu04", "DMU/dmu05", "DMU/dmu41", "DMU/dmu42",
    #                       "DMU/dmu43", "DMU/dmu44", "DMU/dmu45",
    #                       "TA/ta41", "TA/ta42", "TA/ta43", "TA/ta44", "TA/ta45", "TA/ta46", "TA/ta47", "TA/ta48", "TA/ta49", "TA/ta50",
    #                       "DMU/dmu36", "DMU/dmu37", "DMU/dmu38", "DMU/dmu39", "DMU/dmu40", "DMU/dmu76", "DMU/dmu77",
    #                       "DMU/dmu78", "DMU/dmu79", "DMU/dmu80",
    #                       ]
    #     result_dict = {}
    #     op_list = ["mean","std"]
    #     index_list = []
    #     detail_res_list = {}
    #     for benchmark in benchmark_list:
    #         for std in proctime_std:
    #             for rule in dispatching_rules_list:
    #                 print("benchmark_name:",benchmark,",time std:",std,",dispatching rule:",rule)
    #                 makespans = []
    #                 for idx in range(20):
    #                     makespans.append(test_on_single_instance(benchname=benchmark,
    #                                                              disrule_name=rule,
    #                                                              proctime_std=std,
    #                                                              proc_seed=idx+10,
    #                                                              sched_ratio=None))
    #                 print(makespans)
    #                 detail_res_list[f'{benchmark}-{std}-{rule}'] = makespans
    #                 if rule not in result_dict.keys():
    #                     result_dict[rule] = []
    #                 # mean_mp, max_mp, min_mp = np.mean(makespans), np.max(makespans), np.min(makespans)
    #                 mean_mp, std_mp = np.mean(makespans), np.std(makespans)
    #                 result_dict[rule].append(mean_mp)
    #                 result_dict[rule].append(std_mp)
    #                 # result_dict[rule].append(min_mp)
    #                 print(mean_mp,'\n', std_mp)
    #             for val in op_list:
    #                 index_list.append(f'{benchmark}-{std}-{val}')
    #
    #     if output_flag_2:
    #         makespan_data_pd = pd.DataFrame(result_dict, index=index_list)
    #         writer = pd.ExcelWriter(f'experiment_2 result {TIMESTAMP}.xlsx')
    #         makespan_data_pd.to_excel(writer,float_format='%.3f')
    #         writer.save()
    #         print("experiment_2 rules result output success")
    #
    #         makespan_data_pd = pd.DataFrame(detail_res_list, index=list(range(1,21)))
    #         writer = pd.ExcelWriter(f'experiment_2 detailed result {TIMESTAMP}.xlsx')
    #         makespan_data_pd.to_excel(writer,float_format='%.3f')
    #         writer.save()
    #         print("experiment_2 rules detail result output success")

    # experiment_3: test on all benchmark with machine breakdown
    from pyjssp.utils import Timer

    Timer_exp_mb = Timer("Dynamic experiment with machine breakdown")
    with Timer_exp_mb:
        np.random.seed(configs.np_seed_validation)
        random.seed(configs.python_seed)
        output_flag_3 = True
        dispatching_rules_list = [None, "FIFO", "LIFO", "SPT", "LPT", "STPT", "LTPT", "LOR", "MOR", "MWKR", "FDD/MWKR"]
        # benchmark_list = ["FT/ft06", "LA/la16","DMU/dmu06","TA/ta61"]
        Ag_list = [0.03, 0.05, 0.08]
        benchmark_list = ["LA/la16", "LA/la17", "LA/la18", "LA/la19", "LA/la20",
                          "DMU/dmu01", "DMU/dmu02", "DMU/dmu03", "DMU/dmu04", "DMU/dmu05", "DMU/dmu41", "DMU/dmu42",
                          "DMU/dmu43", "DMU/dmu44", "DMU/dmu45",
                          "TA/ta41", "TA/ta42", "TA/ta43", "TA/ta44", "TA/ta45", "TA/ta46", "TA/ta47", "TA/ta48",
                          "TA/ta49", "TA/ta50",
                          "DMU/dmu36", "DMU/dmu37", "DMU/dmu38", "DMU/dmu39", "DMU/dmu40", "DMU/dmu76", "DMU/dmu77",
                          "DMU/dmu78", "DMU/dmu79", "DMU/dmu80",
                          ]

        result_dict = {}
        op_list = ["mean","std"]
        index_list = []
        detail_res_list = {}
        for benchmark in benchmark_list:
            for Ag in Ag_list:
                for rule in dispatching_rules_list:
                    makespans = []
                    print("benchmark_name:",benchmark,",bk Ag:",Ag,",dispatching rule:",rule)
                    for idx in range(20):
                        makespans.append(test_on_single_instance(benchname=benchmark,
                                                                 disrule_name=rule,
                                                                 mbrk_Ag=Ag,
                                                                 mbrk_seed=idx+10,
                                                                 sched_ratio=None))
                    print(makespans)
                    detail_res_list[f'{benchmark}-{Ag}-{rule}'] = makespans
                    if rule not in result_dict.keys():
                        result_dict[rule] = []
                    mean_mp, std_mp = np.mean(makespans), np.std(makespans)
                    result_dict[rule].append(mean_mp)
                    result_dict[rule].append(std_mp)

                    print(mean_mp,'\n', std_mp)
                for val in op_list:
                    index_list.append(f'{benchmark}-{Ag}-{val}')

        if output_flag_3:
            makespan_data_pd = pd.DataFrame(result_dict, index=index_list)
            writer = pd.ExcelWriter(f'experiment_3 bk rules result {TIMESTAMP}.xlsx')
            makespan_data_pd.to_excel(writer,float_format='%.3f')
            writer.save()
            print("experiment_3 bk rules result output success")

            makespan_data_pd = pd.DataFrame(detail_res_list, index=list(range(1,21)))
            writer = pd.ExcelWriter(f'experiment_3 bk rules detailed result {TIMESTAMP}.xlsx')
            makespan_data_pd.to_excel(writer,float_format='%.3f')
            writer.save()
            print("experiment_3 bk rules result output success")

        # experiment_4: test on all benchmark with machine breakdown
        # output_flag_4 = True
        # dispatching_rules_list = [None, "FIFO", "LIFO", "SPT", "LPT", "STPT", "LTPT", "LOR", "MOR", "MWKR", "FDD/MWKR"]
        # # dispatching_rules_list = ["FIFO"]
        # benchmark_list = ["FT/ft20", "SWV/swv20","TA/ta60","TA/ta80"]
        # op_list = ["mean","max","min"]
        # index_list = []
        # result_dict = {}
        # for benchmark in benchmark_list:
        #     for rule in dispatching_rules_list:
        #         print("benchmark_name:",benchmark,"rule name:",rule,"Ag=0.05")
        #         makespans = []
        #         for idx in range(50):
        #             makespans.append(test_on_single_instance(benchname=benchmark,
        #                             disrule_name=rule,sched_ratio=None,mbrk_Ag=0.05,mbrk_seed=idx+1))
        #             #makespan = test_on_single_instance(benchname="dmu06",disrule_name=rule,sched_ratio=None,proctime_std=3)
        #         print(makespans)
        #         if rule not in result_dict.keys():
        #             result_dict[rule] = []
        #         mean_mp, max_mp, min_mp = np.mean(makespans), np.max(makespans), np.min(makespans)
        #         result_dict[rule].append(mean_mp)
        #         result_dict[rule].append(max_mp)
        #         result_dict[rule].append(min_mp)
        #         print(mean_mp,'\n',max_mp,'\n',min_mp)
        #     for val in op_list:
        #         index_list.append(f'{benchmark}-{val}')
        #
        # if output_flag_4:
        #     makespan_data_pd = pd.DataFrame(result_dict, index=index_list)
        #     writer = pd.ExcelWriter(f'experiment_4 result {TIMESTAMP}.xlsx')
        #     makespan_data_pd.to_excel(writer,float_format='%.3f')
        #     writer.save()
        #     print("experiment_4 result output success")
    from pyjssp.utils import Timer

    Timer_exp_mb_and_std = Timer("Dynamic experiment with machine breakdown")
    with Timer_exp_mb_and_std:
        np.random.seed(configs.np_seed_validation)
        random.seed(configs.python_seed)
        output_flag_3 = True
        dispatching_rules_list = [None, "FIFO", "LIFO", "SPT", "LPT", "STPT", "LTPT", "LOR", "MOR", "MWKR",
                                  "FDD/MWKR"]
        # benchmark_list = ["FT/ft06", "LA/la16","DMU/dmu06","TA/ta61"]
        Ag = 0.05
        std = 2
        benchmark_list = ["LA/la16", "LA/la17", "LA/la18", "LA/la19", "LA/la20",
                          "DMU/dmu01", "DMU/dmu02", "DMU/dmu03", "DMU/dmu04", "DMU/dmu05", "DMU/dmu41", "DMU/dmu42",
                          "DMU/dmu43", "DMU/dmu44", "DMU/dmu45",
                          "TA/ta41", "TA/ta42", "TA/ta43", "TA/ta44", "TA/ta45", "TA/ta46", "TA/ta47", "TA/ta48",
                          "TA/ta49", "TA/ta50",
                          "DMU/dmu36", "DMU/dmu37", "DMU/dmu38", "DMU/dmu39", "DMU/dmu40", "DMU/dmu76", "DMU/dmu77",
                          "DMU/dmu78", "DMU/dmu79", "DMU/dmu80",
                          ]

        result_dict = {}
        op_list = ["mean", "std"]
        index_list = []
        detail_res_list = {}
        for benchmark in benchmark_list:
            for rule in dispatching_rules_list:
                makespans = []
                print("benchmark_name:", benchmark, ",bk Ag:", Ag, "proc std:", std ,",dispatching rule:", rule)
                for idx in range(20):
                    makespans.append(test_on_single_instance(benchname=benchmark,
                                                             disrule_name=rule,
                                                             mbrk_Ag=Ag,
                                                             proctime_std= std,
                                                             mbrk_seed=idx + 10,
                                                             proc_seed=idx + 10,
                                                             sched_ratio=None))
                print(makespans)
                detail_res_list[f'{benchmark}-{Ag}-{rule}'] = makespans
                if rule not in result_dict.keys():
                    result_dict[rule] = []
                mean_mp, std_mp = np.mean(makespans), np.std(makespans)
                result_dict[rule].append(mean_mp)
                result_dict[rule].append(std_mp)

                print(mean_mp, '\n', std_mp)
            for val in op_list:
                index_list.append(f'{benchmark}-{Ag}-{val}')

        if output_flag_3:
            makespan_data_pd = pd.DataFrame(result_dict, index=index_list)
            writer = pd.ExcelWriter(f'experiment_5 bk and std rules result {TIMESTAMP}.xlsx')
            makespan_data_pd.to_excel(writer, float_format='%.3f')
            writer.save()
            print("experiment_5 bk and std rules result output success")

            makespan_data_pd = pd.DataFrame(detail_res_list, index=list(range(1, 21)))
            writer = pd.ExcelWriter(f'experiment_5 bk and std rules detailed result {TIMESTAMP}.xlsx')
            makespan_data_pd.to_excel(writer, float_format='%.3f')
            writer.save()
            print("experiment_5 bk and std rules result output success")