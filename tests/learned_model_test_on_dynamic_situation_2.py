import random
import numpy as np
import pandas as pd

from Params import configs
# from validation import validate
from pyjssp.simulators import JSSPSimulator
import time
import os
import torch
from PPO_jssp_multiInstances_update import PPO
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

def test_on_single_instance(benchname,model=None,proctime_std=0,proc_seed=None,sched_ratio=None,mbrk_Ag=None,mbrk_seed=None):
    env = JSSPSimulator(num_jobs=None, num_machines=None)
    fea, adj, _, reward, candidate, mask,done = env.reset(jssp_path=parent_path + f'/benchmark/{benchname}.txt',
                                                          proctime_std=proctime_std,
                                                          proc_seed=proc_seed,
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
    TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
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
    # path = './SavedNetwork/{}.pth'.format("-n_j-10-n_m-10--same size training by ES-03-12-11-10")
    # ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    # experiment_1: test on all benchmark
    timer = Timer("test on dynamic experiment")

    with timer:
        output_flag_2 = True
        model_list = ['-n_j-10-n_m-10-training on static env without rem_op_fea05-13-16-18',]
        # model_list = ["-n_j-10-n_m-10-training on static env without com_fea05-11-22-06"]
        # model_list = ['-n_j-10-n_m-10-training on static env without prt_fea05-10-15-43']
        # model_list = ['-n_j-10-n_m-10-training on static env without wait_time_fea05-09-10-10']
        # model_list = ['-n_j-10-n_m-10-training on static without node_status_fea05-07-21-35']
        # model_list = ['-n_j-10-n_m-10-training on static env with 5 features05-06-11-02']

        proctime_std = [1, 2, 3]
        benchmark_list = ["LA/la16", "LA/la17", "LA/la18", "LA/la19", "LA/la20",
                          "DMU/dmu01", "DMU/dmu02", "DMU/dmu03", "DMU/dmu04", "DMU/dmu05",
                          "TA/ta41", "TA/ta42", "TA/ta43", "TA/ta44", "TA/ta45",
                          "DMU/dmu36", "DMU/dmu37", "DMU/dmu38", "DMU/dmu39", "DMU/dmu40",
                          ]
        result_dict = {}
        op_list = ["mean"]
        index_list = []
        detail_res_list = {}
        model_T = None
        for benchmark in benchmark_list:
            for std in proctime_std:
                for model in model_list:
                    model_T = model
                    path = './SavedNetwork/{}.pth'.format(model)
                    ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                    print("benchmark_name:", benchmark, ",proctime_std:", std, ",model:", model)
                    makespans = []
                    for idx in range(20):
                        makespans.append(test_on_single_instance(benchname=benchmark,
                                                                 model=ppo,
                                                                 proctime_std=std,
                                                                 proc_seed=idx + 10))
                    print(makespans)
                    detail_res_list[f'{benchmark}-{std}-{model}'] = makespans
                    if model not in result_dict.keys():
                        result_dict[model] = []
                    mean_mp, std_mp = np.mean(makespans), np.std(makespans)
                    result_dict[model].append(mean_mp)
                    # result_dict[model].append(std_mp)
                    print(mean_mp,'\n',std_mp)
                for val in op_list:
                    index_list.append(f'{benchmark}-{std}-{val}')

        if output_flag_2:
            makespan_data_pd = pd.DataFrame(result_dict, index=index_list)
            writer = pd.ExcelWriter(f'Excel_save_files/xiaorongshiyan/'
                                    f'experiment_2 test with model result {TIMESTAMP}-{model_T}.xlsx')
            makespan_data_pd.to_excel(writer, float_format='%.3f')
            writer.save()
            print("experiment_2 test with model result output success")

            makespan_data_pd = pd.DataFrame(detail_res_list, index=list(range(1, 21)))
            writer = pd.ExcelWriter(f'Excel_save_files/xiaorongshiyan/'
                                    f'experiment_2 test with model detailed result {TIMESTAMP}-{model_T}.xlsx')
            makespan_data_pd.to_excel(writer, float_format='%.3f')
            writer.save()
            print("experiment_2 test with model detail result output success")
            print("Dynamic experiment by model with stochastic processing time result output success")

    # experiment_4: test on all benchmark with machine breakdown
    from pyjssp.utils import Timer
    timer = Timer("test on bk dynamic experiment")

    with timer:
        output_flag_4 = True
        model_list = ['-n_j-10-n_m-10-training on static env without rem_op_fea05-13-16-18',]
        # model_list = ["-n_j-10-n_m-10-training on static env without com_fea05-11-22-06"]
        # model_list = ['-n_j-10-n_m-10-training on static env without prt_fea05-10-15-43']
        # model_list = ['-n_j-10-n_m-10-training on static env without wait_time_fea05-09-10-10']
        # model_list = ['-n_j-10-n_m-10-training on static without node_status_fea05-07-21-35']
        # model_list = ['-n_j-10-n_m-10-training on static env with 5 features05-06-11-02']
        Ag_list = [0.03, 0.05, 0.08]
        benchmark_list = ["LA/la16", "LA/la17", "LA/la18", "LA/la19", "LA/la20",
                          "DMU/dmu01", "DMU/dmu02", "DMU/dmu03", "DMU/dmu04", "DMU/dmu05",
                          "TA/ta41", "TA/ta42", "TA/ta43", "TA/ta44", "TA/ta45",
                          "DMU/dmu36", "DMU/dmu37", "DMU/dmu38", "DMU/dmu39", "DMU/dmu40",
                          ]
        result_dict = {}
        op_list = ["mean"]
        index_list = []
        detail_res_list = {}
        model_T = None
        for benchmark in benchmark_list:
            for Ag in Ag_list:
                for model in model_list:
                    model_T = model
                    path = './SavedNetwork/{}.pth'.format(model)
                    ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                    print("benchmark_name:", benchmark, ",breakdown Ag:", Ag, ",model:", model)
                    makespans = []
                    for idx in range(20):
                        makespans.append(test_on_single_instance(benchname=benchmark,
                                                                 model=ppo,
                                                                 mbrk_Ag=Ag,
                                                                 mbrk_seed=idx + 10))
                    print(makespans)
                    detail_res_list[f'{benchmark}-{Ag}-{model}'] = makespans
                    if model not in result_dict.keys():
                        result_dict[model] = []
                    mean_mp, std_mp = np.mean(makespans), np.std(makespans)
                    result_dict[model].append(mean_mp)
                    # result_dict[model].append(std_mp)
                    print(mean_mp,'\n',std_mp)
                for val in op_list:
                    index_list.append(f'{benchmark}-{Ag}-{val}')

        if output_flag_4:
            makespan_data_pd = pd.DataFrame(result_dict, index=index_list)
            writer = pd.ExcelWriter(f'Excel_save_files/xiaorongshiyan/'
                                    f'experiment_4 bk test with model result {TIMESTAMP}-{model_T}.xlsx')
            makespan_data_pd.to_excel(writer, float_format='%.3f')
            writer.save()
            print("experiment_4 bk test with model result output success")

            makespan_data_pd = pd.DataFrame(detail_res_list, index=list(range(1, 21)))
            writer = pd.ExcelWriter(f'Excel_save_files/xiaorongshiyan/'
                                    f'experiment_4 bk test with model detailed result {TIMESTAMP}-{model_T}.xlsx')
            makespan_data_pd.to_excel(writer, float_format='%.3f')
            writer.save()
            print("experiment_4 bk test with model detail result output success")
    # from pyjssp.utils import Timer
    # timer = Timer("test on bk dynamic experiment")

    # with timer:
    #     output_flag_5 = True
    #     model_list = ['-n_j-10-n_m-10-training on static env04-10-21-19',
    #                   '-n_j-10-n_m-10-training on static env 0410 again04-24-16-24']
    #     # Ag_list = [0.03, 0.05, 0.08]
    #     Ag = 0.05
    #     std = 2
    #     benchmark_list = ["LA/la16", "LA/la17", "LA/la18", "LA/la19", "LA/la20",
    #                       "DMU/dmu01", "DMU/dmu02", "DMU/dmu03", "DMU/dmu04", "DMU/dmu05",
    #                       "TA/ta41", "TA/ta42", "TA/ta43", "TA/ta44", "TA/ta45",
    #                       "DMU/dmu36", "DMU/dmu37", "DMU/dmu38", "DMU/dmu39", "DMU/dmu40",
    #                       ]
    #     result_dict = {}
    #     op_list = ["mean"]
    #     index_list = []
    #     detail_res_list = {}
    #     for benchmark in benchmark_list:
    #         for model in model_list:
    #             path = './SavedNetwork/{}.pth'.format(model)
    #             ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    #             print("benchmark_name:", benchmark, ",breakdown Ag:", Ag, ",model:", model)
    #             makespans = []
    #             for idx in range(20):
    #                 makespans.append(test_on_single_instance(benchname=benchmark,
    #                                                          model=ppo,
    #                                                          mbrk_Ag=Ag,
    #                                                          proctime_std=std,
    #                                                          proc_seed=idx + 10,
    #                                                          mbrk_seed=idx + 10))
    #             print(makespans)
    #             detail_res_list[f'{benchmark}-{Ag}-{model}'] = makespans
    #             if model not in result_dict.keys():
    #                 result_dict[model] = []
    #             mean_mp, std_mp = np.mean(makespans), np.std(makespans)
    #             result_dict[model].append(mean_mp)
    #             # result_dict[model].append(std_mp)
    #             print(mean_mp, '\n', std_mp)
    #         for val in op_list:
    #             index_list.append(f'{benchmark}-{Ag}-{val}')
    #
    #     if output_flag_5:
    #         makespan_data_pd = pd.DataFrame(result_dict, index=index_list)
    #         writer = pd.ExcelWriter(f'experiment_5 bk test with model result {TIMESTAMP}.xlsx')
    #         makespan_data_pd.to_excel(writer, float_format='%.3f')
    #         writer.save()
    #         print("experiment_5 bk test with model result output success")
    #
    #         makespan_data_pd = pd.DataFrame(detail_res_list, index=list(range(1, 21)))
    #         writer = pd.ExcelWriter(f'experiment_5 bk test with model detailed result {TIMESTAMP}.xlsx')
    #         makespan_data_pd.to_excel(writer, float_format='%.3f')
    #         writer.save()
    #         print("experiment_5 bk test with model detail result output success")


