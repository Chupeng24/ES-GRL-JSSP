from mb_agg import *
from agent_utils import *
import numpy as np
from Params import configs
import argparse
import torch
import time
from os import path

device = torch.device(configs.device)
np.random.seed(configs.np_seed_train)
d = path.dirname(__file__)
parent_path = path.dirname(d) #获得d所在的目录,即d的父级目录

parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
parser.add_argument('--Pn_j', type=int, default=6, help='Number of jobs of instances to test')
parser.add_argument('--Pn_m', type=int, default=6, help='Number of machines instances to test')
parser.add_argument('--Nn_j', type=int, default=6, help='Number of jobs on which to be loaded net are trained')
parser.add_argument('--Nn_m', type=int, default=6, help='Number of machines on which to be loaded net are trained')
parser.add_argument('--low', type=int, default=1, help='LB of duration')
parser.add_argument('--high', type=int, default=99, help='UB of duration')
parser.add_argument('--seed', type=int, default=200, help='Seed for validate set generation')
parser.add_argument('--sched_ratio', type=int, default=0.2, help='Seed for validate set generation')
params = parser.parse_args()

N_JOBS_P = params.Pn_j
N_MACHINES_P = params.Pn_m
LOW = params.low
HIGH = params.high
SEED = params.seed
N_JOBS_N = params.Nn_j
N_MACHINES_N = params.Nn_m
Sched_ratio = params.sched_ratio
disrule_name = "SPT"


from pyjssp.simulators import JSSPSimulator
from tests.PPO_jssp_multiInstances import PPO
env = JSSPSimulator(num_jobs=None, num_machines=None)

ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
          n_j=N_JOBS_P,
          n_m=N_MACHINES_P,
          num_layers=configs.num_layers,
          neighbor_pooling_type=configs.neighbor_pooling_type,
          input_dim=configs.input_dim,
          hidden_dim=configs.hidden_dim,
          num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
          num_mlp_layers_actor=configs.num_mlp_layers_actor,
          hidden_dim_actor=configs.hidden_dim_actor,
          num_mlp_layers_critic=configs.num_mlp_layers_critic,
          hidden_dim_critic=configs.hidden_dim_critic)
path = parent_path +  './SavedNetwork/6_6_1_99_2021-12-15-22-33-35.pth'
# ppo.policy.load_state_dict(torch.load(path))
ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
# ppo.policy.eval()
g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                         batch_size=torch.Size([1, N_JOBS_N*N_MACHINES_N, N_JOBS_N*N_MACHINES_N]),
                         n_nodes=N_JOBS_N*N_MACHINES_N,
                         device=device)
makespans = []
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
    dataLoaded = np.load(parent_path + './DataGen/generatedData' + str(configs.n_j) + '_' + str(configs.n_m) + '_Seed' + str(configs.np_seed_validation) + '.npy')
    vali_data = []
    for i in range(dataLoaded.shape[0]):
        vali_data.append((dataLoaded[i][0], dataLoaded[i][1]))

    for index,data in enumerate(vali_data):
        fea, adj, _, _, candidate, undoable_mask, done \
            = env.reset(processing_time_matrix=data[0], machine_matrix=data[1])
        start_time = time.time()
        while True:
            # _, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name)
            fea_tensor = torch.from_numpy(fea).to(device)
            adj_tensor = torch.from_numpy(adj).to(device)
            candidate_tensor = torch.from_numpy(candidate).to(device)
            mask_tensor = torch.from_numpy(undoable_mask).to(device)

            pi, _ = ppo.policy(x=fea_tensor,
                               graph_pool=g_pool_step,
                               padded_nei=None,
                               adj=adj_tensor,
                               candidate=candidate_tensor.unsqueeze(0),
                               mask=mask_tensor.unsqueeze(0))
            action = greedy_select_action(pi, candidate)
            fea, adj, _, _, candidate, undoable_mask, done = env.step(action)


            if done:
                print("makespan: ",env.global_time)
                makespans.append(env.global_time)
                end_time = time.time()
                interval = end_time - start_time
                print("calculate time:",interval)
                # env.draw_gantt_chart("model_test_result.html","GenData_6x6_index"+str(index),100)
                break

print(np.mean(makespans))