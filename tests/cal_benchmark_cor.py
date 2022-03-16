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

def obatin_instance_procsum():
    path_list = []
    name_list = []
    instance_procsum = []
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
        env.reset(jssp_path=elem)
        # env.reset(jssp_path='FT10.txt',proctime_std=0)
        # start_time = time.time()
        instance_procsum.append(np.sum(env.processing_time_matrix))
    return instance_procsum

if __name__ == '__main__':
    name_list = []
    for root, dirs, files in os.walk(parent_path + f'/benchmark', topdown=False):
        for name in files:
            # print(name.split(".",1)[0])
            name_list.append(name.split(".",1)[0])

    instance_procsum = obatin_instance_procsum()

    cor_data_pd = pd.DataFrame(instance_procsum, index=name_list)
    writer = pd.ExcelWriter(f'cor of benchmark proctime sum and makespan.xlsx')
    cor_data_pd.to_excel(writer,float_format='%.3f')
    writer.save()
    print("done")

