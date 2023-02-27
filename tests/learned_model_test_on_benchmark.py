import random
import numpy as np
import pandas as pd

from Params import configs
# from validation import validate
from pyjssp.simulators import JSSPSimulator
import time
import os
import torch
from PPO_train import PPO
from mb_agg import *
from agent_utils import *
from pyjssp.utils import *

#from os import path
import profile
SEED = 200
np.random.seed(configs.np_seed_validation)
random.seed(SEED)
torch.manual_seed(configs.torch_seed)
d = os.path.dirname(__file__)
parent_path = os.path.dirname(d) #获得d所在的目录,即d的父级目录

device = torch.device(configs.device)

#@profile
def test_on_all_instance(model=None,proctime_std=0,sched_ratio=None):
    path_list = []
    name_list = []
    count = 0
    makespans = []
    for root, dirs, files in os.walk(parent_path + f'/benchmark', topdown=False):
        for name in files:
            count += 1
            name_list.append(name)
            path_list.append(os.path.join(root, name))
    print("instance count:",count)
    env = JSSPSimulator(num_jobs=None, num_machines=None)
    for elem in path_list:
        fea, adj, _, reward, candidate, mask,done = env.reset(jssp_path=elem,proctime_std=proctime_std,sched_ratio=sched_ratio)
        g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                                 batch_size=torch.Size([1, env.num_ops, env.num_ops]),
                                 n_nodes=env.num_ops,
                                 device=device)
        # env.reset(jssp_path='FT10.txt',proctime_std=0)
        start_time = time.time()
        while True:
            if env.random_stop_flag == True:
                fea_tensor = torch.from_numpy(fea).to(device)
                adj_tensor = torch.from_numpy(adj).to(device)
                candidate_tensor = torch.from_numpy(candidate).to(device)
                mask_tensor = torch.from_numpy(mask).to(device)
                with torch.no_grad():
                    pi, _ = model.policy(x=fea_tensor,
                                         graph_pool=g_pool_step,
                                         n_j = env.num_jobs,
                                         padded_nei=None,
                                         adj=adj_tensor,
                                         candidate=candidate_tensor.unsqueeze(0),
                                         mask=mask_tensor.unsqueeze(0))
                action = greedy_select_action(pi, candidate)
                fea, adj, _, reward, candidate, mask,done = env.step(action=action.item(),disrule_name=None)
            else:
                _, _, _, _, _,_, done = env.step(action=None,disrule_name=None)
            if done:
                print(env.global_time)
                end_time = time.time()
                interval = end_time - start_time
                makespans.append(env.global_time)
                break
    #print("mean_makespan:",np.mean(makespan_list))

def test_on_single_instance(benchname,model=None,proctime_std=0,sched_ratio=None,mbrk_Ag=0.0,mbrk_seed=None):
    env = JSSPSimulator(num_jobs=None, num_machines=None)
    fea, adj, _, reward, candidate, mask,done = env.reset(jssp_path=parent_path + f'/benchmark/{benchname}.txt',
                                                          proctime_std=proctime_std,
                                                          sched_ratio=sched_ratio,
                                                          mbrk_Ag=mbrk_Ag,
                                                          mbrk_seed=mbrk_seed)
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1, env.num_ops, env.num_ops]),
                             n_nodes=env.num_ops,
                             device=device)
    # env.reset(jssp_path='FT10.txt',proctime_std=0)
    start_time = time.time()
    while True:
        if env.random_stop_flag == True:
            fea_tensor = torch.from_numpy(fea).to(device)
            adj_tensor = torch.from_numpy(adj).to(device)
            candidate_tensor = torch.from_numpy(candidate).to(device)
            mask_tensor = torch.from_numpy(mask).to(device)
            with torch.no_grad():
                pi, _ = model.policy(x=fea_tensor,
                                     graph_pool=g_pool_step,
                                     n_j = env.num_jobs,
                                     padded_nei=None,
                                     adj=adj_tensor,
                                     candidate=candidate_tensor.unsqueeze(0),
                                     mask=mask_tensor.unsqueeze(0))
            action = greedy_select_action(pi, candidate)
            fea, adj, _, reward, candidate, mask,done = env.step(action=action.item())
        else:
            fea, adj, _, reward, candidate, mask,done = env.step(action=None,disrule_name=None)
        if done:
            # print(env.global_time)
            end_time = time.time()
            interval = end_time - start_time
            return env.global_time



if __name__ == '__main__':

    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=None,
              n_m=None,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)
    path = './SavedNetwork/{}.pth'.format("-n_j-10-n_m-10-ES training on static env03-22-22-20")
    ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    # experiment_1: test on all benchmark
    timer = Timer("experiment_1: test on all benchmark")

    with timer:
        test_on_all_instance(model=ppo)

    # print("experiment_1 result output success")
    #
    # # experiment_2: test on all benchmark with stochastic processing time
    # np.random.seed(configs.np_seed_validation)
    # random.seed(configs.python_seed)
    # benchmark_list = ["FT/ft06", "LA/la16","DMU/dmu06","TA/ta61"]
    # proctime_std = [0, 1, 2, 3]
    # result_dict = {}
    # for benchmark in benchmark_list:
    #     for std in proctime_std:
    #         print("benchmark_name:",benchmark,",time std:",std)
    #         makespans = []
    #         for idx in range(50):
    #             makespans.append(test_on_single_instance(benchname=benchmark,
    #                                                      model=ppo,proctime_std=std,sched_ratio=None))
    #         print(makespans)
    #         mean_mp, max_mp, min_mp = np.mean(makespans), np.max(makespans), np.min(makespans)
    #         print(np.mean(makespans),'\n',np.max(makespans),'\n',np.min(makespans))
    # print("experiment_2 result output success")
    #
    # # experiment_3: test on all benchmark with non-zero states
    # np.random.seed(configs.np_seed_validation)
    # random.seed(configs.python_seed)
    # benchmark_list = ["FT/ft06", "LA/la16","DMU/dmu06","TA/ta61"]
    # sched_ratio = [0, 0.2, 0.4, 0.6]
    # index_list = ["median", "max", "min"]
    # detail_res_list = {}
    # for benchmark in benchmark_list:
    #     for ratio in sched_ratio:
    #         makespans = []
    #         print("benchmark_name:",benchmark,",schedule ratio:",ratio)
    #         for idx in range(50):
    #             makespans.append(test_on_single_instance(benchname=benchmark,model=ppo,sched_ratio=ratio))
    #         print(makespans)
    #         median_mp, max_mp, min_mp = np.median(makespans), np.max(makespans), np.min(makespans)
    #         print(median_mp,'\n',max_mp,'\n',min_mp)
    # print("experiment_3 result output success")
    #
    # # experiment_4: test on all benchmark with machine breakdown
    # np.random.seed(configs.np_seed_validation)
    # random.seed(configs.python_seed)
    # benchmark_list = ["FT/ft20", "SWV/swv20","TA/ta60","TA/ta80"]
    # for benchmark in benchmark_list:
    #         print("benchmark_name:",benchmark,"Ag=0.05")
    #         makespans = []
    #         for idx in range(50):
    #             makespans.append(test_on_single_instance(benchname=benchmark,
    #                                                      model=ppo,sched_ratio=None,mbrk_Ag=0.05,mbrk_seed=idx))
    #             #makespan = test_on_single_instance(benchname="dmu06",disrule_name=rule,sched_ratio=None,proctime_std=3)
    #         print(makespans)
    #         mean_mp, max_mp, min_mp = np.mean(makespans), np.max(makespans), np.min(makespans)
    #         print(mean_mp,'\n',max_mp,'\n',min_mp)
    # print("experiment_4 result output success")
