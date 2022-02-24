import time

def validate(vali_set,disrule_name=None):
    N_JOBS = vali_set[0][0].shape[0]
    N_MACHINES = vali_set[0][0].shape[1]
    from pyjssp.simulators import JSSPSimulator

    import numpy as np
    import torch
    from Params import configs


    device = torch.device(configs.device)

    make_spans = []
    cal_time = []
    # rollout using SavedNetwork
    for data in vali_set:
        proctime_matrix = data[0]
        m_matrix = data[1]
        env = JSSPSimulator(num_jobs=N_JOBS, num_machines=N_MACHINES)
        env.reset(machine_matrix=m_matrix,processing_time_matrix=proctime_matrix)
        start_time = time.time()
        while True:
            _, _, _, _, _,_, done = env.step(action=None,disrule_name=disrule_name)
            # for n in g.nodes():
            #     print('{}:'.format(n))
            #     print(g.nodes[n])
            # s.plot_graph()
            # print("###########################################")
            if done:
                # print(env.global_time)
                # s.plot_graph()
                # s.job_manager.draw_gantt_chart("random_test_result.html","Ft06 Scheduling result",100)
                # for m_id,m in env.machine_manager.machines.items():
                #     print(m_id,"#########################")
                #     for done_op in m.done_ops:
                #         print(done_op.sur_id)
                make_spans.append(env.global_time)

                end_time = time.time()
                interval = end_time - start_time
                cal_time.append(interval)
                break
    return np.array(make_spans), np.array(cal_time)


# if __name__ == '__main__':
#
#     # from uniform_instance_gen import uni_instance_gen
#     # import numpy as np
#     # import time
#     # import argparse
#     # #from Params import configs
#     #
#     # parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
#     # parser.add_argument('--Pn_j', type=int, default=15, help='Number of jobs of instances to test')
#     # parser.add_argument('--Pn_m', type=int, default=15, help='Number of machines instances to test')
#     # parser.add_argument('--Nn_j', type=int, default=15, help='Number of jobs on which to be loaded net are trained')
#     # parser.add_argument('--Nn_m', type=int, default=15, help='Number of machines on which to be loaded net are trained')
#     # parser.add_argument('--low', type=int, default=1, help='LB of duration')
#     # parser.add_argument('--high', type=int, default=99, help='UB of duration')
#     # parser.add_argument('--seed', type=int, default=200, help='Cap seed for validate set generation')
#     # parser.add_argument('--n_vali', type=int, default=100, help='validation set size')
#     # params = parser.parse_args()
#     #
#     # N_JOBS_P = params.Pn_j
#     # N_MACHINES_P = params.Pn_m
#     # LOW = params.low
#     # HIGH = params.high
#     # N_JOBS_N = params.Nn_j
#     # N_MACHINES_N = params.Nn_m
#     #
#     # # from PPO_jssp_multiInstances import PPO
#     # import torch
#     #
#     # # ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
#     # #           n_j=N_JOBS_P,
#     # #           n_m=N_MACHINES_P,
#     # #           num_layers=configs.num_layers,
#     # #           neighbor_pooling_type=configs.neighbor_pooling_type,
#     # #           input_dim=configs.input_dim,
#     # #           hidden_dim=configs.hidden_dim,
#     # #           num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
#     # #           num_mlp_layers_actor=configs.num_mlp_layers_actor,
#     # #           hidden_dim_actor=configs.hidden_dim_actor,
#     # #           num_mlp_layers_critic=configs.num_mlp_layers_critic,
#     # #           hidden_dim_critic=configs.hidden_dim_critic)
#     #
#     # # path = './{}.pth'.format(str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_' + str(LOW) + '_' + str(HIGH))
#     # # ppo.policy.load_state_dict(torch.load(path))
#     # # 这里是试验不同的随机种子的情况
#     # SEEDs = range(0, params.seed, 10)
#     # result = []
#     # for SEED in SEEDs:
#     #
#     #     np.random.seed(SEED)
#     #
#     #     vali_data = [uni_instance_gen(n_j=N_JOBS_P, n_m=N_MACHINES_P, low=LOW, high=HIGH) for _ in range(params.n_vali)]
#     #
#     #     makespan = - validate(vali_data, ppo.policy)
#     #     print(makespan.mean())
#     #
#     #
#     # # print(min(result))

