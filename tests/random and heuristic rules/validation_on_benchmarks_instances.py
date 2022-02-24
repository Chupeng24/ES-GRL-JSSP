import numpy as np
from Params import configs
from validation import validate
from pyjssp.simulators import JSSPSimulator
from os import path
import time

np.random.seed(configs.np_seed_train)

d = path.dirname(__file__)
parent_path = path.dirname(d) #获得d所在的目录,即d的父级目录

def test_on_gen_instance(benchmark_name,dispatching_rule):
    dataLoaded = np.load(parent_path + './BenchDataNmpy/' + benchmark_name + '.npy')
    print(f"test on {benchmark_name} instance")
    vali_data = []
    for i in range(dataLoaded.shape[0]):
        vali_data.append((dataLoaded[i][0], dataLoaded[i][1]))

    env = JSSPSimulator(num_jobs=configs.n_j, num_machines=configs.n_m)
    disrule_name = dispatching_rule
    make_spans = []
    interval_list =[]

    for data in vali_data:
        proctime_matrix = data[0]
        m_matrix = data[1]
        env.reset(machine_matrix=m_matrix,processing_time_matrix=proctime_matrix)
        start_time = time.time()
        i = 0
        while True:
            _, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name)
            i = i + 1
            if done:
                make_spans.append(env.global_time)

                end_time = time.time()
                interval = end_time - start_time
                interval_list.append(interval)
                break
    print("make_spans:")
    for val in make_spans:
        print(val)
    print("average makespan: ",np.mean(make_spans))
    print("average calculate time:",np.mean(interval_list))
    return np.array(make_spans), np.array(interval_list)

if __name__ == '__main__':
    # 另一种写法
    # dataLoaded = np.load(parent_path + './BenchDataNmpy/' + 'tai15x15' + '.npy')
    # print(f"test on {configs.n_j} x {configs.n_m} instance")
    # vali_data = []
    # for i in range(dataLoaded.shape[0]):
    #     vali_data.append((dataLoaded[i][0], dataLoaded[i][1]))
    # vali_makespan,cal_time = validate(vali_data,disrule_name="SPT")
    # makespan_mean = vali_makespan.mean()
    # cal_time_mean = cal_time.mean()
    # print("average makespan: ",makespan_mean)
    # print("average calculate time:",cal_time_mean)
    ta_instance_names = ["tai15x15","tai20x15","tai20x20","tai30x15","tai30x20","tai50x15","tai50x20","tai100x20"]
    dmu_instance_names = ["dmu20x15","dmu20x20","dmu30x15","dmu30x20","dmu40x15","dmu40x20","dmu50x15","dmu50x20"]

    dispatching_rules_list = ["FIFO", "LIFO", "SPT", "LPT", "STPT", "LTPT", "LOR", "MOR", "MWKR", "FDD/MWKR"]

    for instance_name in ta_instance_names:
        print(f"#####################test on {instance_name} instance#####################")
        for rule in dispatching_rules_list:
            print(f"using dispatching rules {rule}")
            test_on_gen_instance(benchmark_name=instance_name, dispatching_rule=rule)
            print("\n")

    for instance_name in dmu_instance_names:
        print(f"#####################test on {instance_name} instance#####################")
        for rule in dispatching_rules_list:
            print(f"using dispatching rules {rule}")
            test_on_gen_instance(benchmark_name=instance_name, dispatching_rule=rule)
            print("\n")




