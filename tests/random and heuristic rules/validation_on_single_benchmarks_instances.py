import numpy as np
from Params import configs
from validation import validate
from pyjssp.simulators import JSSPSimulator
import time
from os import path
import profile

np.random.seed(configs.np_seed_train)
d = path.dirname(__file__)
parent_path = path.dirname(d) #获得d所在的目录,即d的父级目录

#@profile
def test_on_single_instance():
    env = JSSPSimulator(num_jobs=None, num_machines=None)
    env.reset(jssp_path=parent_path + '/benchmarks/ft06-1.txt',proctime_std=0)
    # env.reset(jssp_path='FT10.txt',proctime_std=0)
    disrule_name = "SPT"
    start_time = time.time()
    i = 0
    while True:
        _, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name)
        i = i + 1
        if done:
            print("makespan: ",env.global_time)
            end_time = time.time()
            interval = end_time - start_time
            print("calculate time:",interval)
            env.draw_gantt_chart("SPT_rule_test_result.html", "Ft06",100)
            break

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

    # env = JSSPSimulator(num_jobs=None, num_machines=None)
    # # env.reset(jssp_path=parent_path + '/benchmarks/FT06.txt',proctime_std=0)
    # env.reset(jssp_path='FT06.txt',proctime_std=0)
    # disrule_name = "SPT"
    # start_time = time.time()
    # while True:
    #     _, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name)
    #
    #     if done:
    #         print("makespan: ",env.global_time)
    #         end_time = time.time()
    #         interval = end_time - start_time
    #         print("calculate time:",interval)
    #         env.draw_gantt_chart("SPT_rule_test_result.html", "Ft06",100)
    #         break
    test_on_single_instance()
    print("end")


    # env.reset(jssp_path=parent_path + '/benchmarks/FT10.txt')
    # start_time = time.time()
    # while True:
    #     _, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name)
    #
    #     if done:
    #         print("makespan: ",env.global_time)
    #         end_time = time.time()
    #         interval = end_time - start_time
    #         print("calculate time:",interval)
    #         break
    # env.reset(jssp_path=parent_path + '/benchmarks/LA11.txt')
    # start_time = time.time()
    # while True:
    #     _, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name)
    #
    #     if done:
    #         print("makespan: ",env.global_time)
    #         end_time = time.time()
    #         interval = end_time - start_time
    #         print("calculate time:",interval)
    #         break