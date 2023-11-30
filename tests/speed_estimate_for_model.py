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
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(configs.torch_seed)
seed_array = np.random.randint(low=0,high=1000000,size=1000000)
global_idx = 0

d = os.path.dirname(__file__)
parent_path = os.path.dirname(d) #获得d所在的目录,即d的父级目录

device = torch.device(configs.device)

def test_on_single_instance(model=None,m_matrix=None,proctime_matrix=None):
    env = JSSPSimulator(num_jobs=None, num_machines=None)
    start_time = time.time()
    fea, adj, _, reward, candidate, mask, done = env.reset(machine_matrix=m_matrix, processing_time_matrix=proctime_matrix)
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1, env.num_ops, env.num_ops]),
                             n_nodes=env.num_ops,
                             device=device)
    # env.reset(jssp_path='FT10.txt',proctime_std=0)
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
            return env.global_time, interval



if __name__ == '__main__':

    from uniform_instance_gen import uni_instance_gen
    import random
    data_generator = uni_instance_gen

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
    path = './SavedNetwork/{}.pth'.format("-n_j-10-n_m-10-training on static env04-10-21-19")
    ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    ###########################################################################
    # Test with fixed num of machines
    # tnum_m = 5
    # list_tnum_n = [5, 10, 15, 20, 25, 30]
    tnum_m = 10
    list_tnum_n = [10, 20, 30, 40, 50, 60]
    for tnum_n in list_tnum_n:
        com_time = []
        for i in range(20):
            np.random.seed(seed_array[global_idx])
            proctime_matrix, m_matrix = data_generator(n_j=tnum_n, n_m=tnum_m, low=1, high=100)
            global_idx = global_idx + 1
            makespan,interval = test_on_single_instance(model=ppo, m_matrix=m_matrix,
                                                        proctime_matrix=proctime_matrix)
            com_time.append(interval)
        print(f"{tnum_n}x{tnum_m}:"," ",np.mean(np.array(com_time)))
        # print(np.mean(np.array(com_time)))
    ############################################################################
    # Test with fixed num of jobs
    # tnum_n = 30
    # list_tnum_m = [5, 10, 15, 20, 25, 30]
    tnum_n = 60
    list_tnum_m = [10, 20, 30, 40, 50, 60]
    for tnum_m in list_tnum_m:
        com_time = []
        for i in range(20):
            np.random.seed(seed_array[global_idx])
            proctime_matrix, m_matrix = data_generator(n_j=tnum_n, n_m=tnum_m, low=1, high=100)
            global_idx = global_idx + 1
            makespan,interval = test_on_single_instance(model=ppo, m_matrix=m_matrix,
                                                        proctime_matrix=proctime_matrix)
            com_time.append(interval)
        print(f"{tnum_n}x{tnum_m}:"," ",np.mean(np.array(com_time)))
        # print(np.mean(np.array(com_time)))







