import numpy as np
from Params import configs
from validation import validate
from pyjssp.simulators import JSSPSimulator
from os import path
import time

np.random.seed(configs.np_seed_train)

d = path.dirname(__file__)
parent_path = path.dirname(d) #获得d所在的目录,即d的父级目录

def test_on_gen_instance(num_m,num_j,dispatching_rule):
    dataLoaded = np.load(parent_path + './DataGen/generatedData' + str(num_j) + '_' + str(num_m) + '_Seed' + str(configs.np_seed_validation) + '.npy')
    print(f"test on {num_j} x {num_m} instance")
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
    print("average makespan: ",np.mean(make_spans))
    print("average calculate time:",np.mean(interval_list))
    return np.array(make_spans), np.array(interval_list)

if __name__ == '__main__':
    # 另一种写法
    # dataLoaded = np.load(parent_path + './DataGen/generatedData' + str(configs.n_j) + '_' + str(configs.n_m) + '_Seed' + str(configs.np_seed_validation) + '.npy')
    # print(f"test on {configs.n_j} x {configs.n_m} instance")
    # vali_data = []
    # for i in range(dataLoaded.shape[0]):
    #     vali_data.append((dataLoaded[i][0], dataLoaded[i][1]))
    # vali_makespan,cal_time = validate(vali_data,disrule_name="SPT")
    # makespan_mean = vali_makespan.mean()
    # cal_time_mean = cal_time.mean()
    # sum_makespan = np.sum(vali_makespan)
    # mean_makespan = sum_makespan/100
    # print("average makespan: ",makespan_mean)
    # print("average calculate time:",cal_time_mean)
    # make_spans,interval_list = test_on_gen_instance(num_m=10, num_j=10, dispatching_rule="LIFO")

    dispatching_rules_list = ["FIFO", "LIFO", "SPT", "LPT", "STPT", "LTPT", "LOR", "MOR", "MWKR", "FDD/MWKR"]
    num_j = 100
    num_m = 20
    for rule in dispatching_rules_list:
        print(f"using dispatching rules {rule}")
        test_on_gen_instance(num_j=num_j, num_m=num_m, dispatching_rule=rule)
        print("\n")
