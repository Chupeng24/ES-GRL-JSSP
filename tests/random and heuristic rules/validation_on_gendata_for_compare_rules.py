import numpy as np
from Params import configs
from validation import validate
from pyjssp.simulators import JSSPSimulator
import time
from os import path

np.random.seed(configs.np_seed_train)
d = path.dirname(__file__)
parent_path = path.dirname(d) #获得d所在的目录,即d的父级目录
make_spans = []

if __name__ == '__main__':
    # dataLoaded = np.load('./BenchDataNmpy/' + 'tai15x15' + '.npy')
    # print(f"test on {configs.n_j} x {configs.n_m} instance")
    # vali_data = []
    # for i in range(dataLoaded.shape[0]):
    #     vali_data.append((dataLoaded[i][0], dataLoaded[i][1]))
    # vali_makespan,cal_time = validate(vali_data,disrule_name="SPT")
    # makespan_mean = vali_makespan.mean()
    # cal_time_mean = cal_time.mean()
    # print("average makespan: ",makespan_mean)
    # print("average calculate time:",cal_time_mean)
    disrule_name = "SPT"
    env = JSSPSimulator(num_jobs=6, num_machines=6)
    dataLoaded = np.load(parent_path + './DataGen/generatedData' + str(6) + '_' + str(6) + '_Seed' + str(200) + '.npy')
    vali_data = []
    for i in range(dataLoaded.shape[0]):
        vali_data.append((dataLoaded[i][0], dataLoaded[i][1]))

    for index,data in enumerate(vali_data):
        env.reset(machine_matrix=data[1],processing_time_matrix=data[0])
        start_time = time.time()
        while True:
            _, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name)

            if done:
                print("makespan: ",env.global_time)
                make_spans.append(env.global_time)
                end_time = time.time()
                interval = end_time - start_time
                print("calculate time:",interval)
                # env.draw_gantt_chart("SPT_rules_test_result.html","GenData_6x6_index"+str(index),100)
                break

print(np.mean(make_spans))
print(np.sum(make_spans))