import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from PPO_jssp_multiInstances_update import PPO
from Params import configs
from uniform_instance_gen import uni_instance_gen
from pyjssp.simulators import JSSPSimulator
import random
import multiprocessing as mp
from validation_update import validate
from agent_utils import *
from mb_agg import *

N_KID = 100                  # half of the training population
N_GENERATION = 5000         # training step
LEARNING_RATE = 0.001       # learning rate
SIGMA = .05                 # mutation strength or step size
N_CORE = mp.cpu_count()-1
MAX_BATCH_EPISODES = 100
NOISE_STD = 0.05
device = torch.device("cpu")

def sample_noise(agent):
    actor_net = agent.policy.actor
    gnn_net = agent.policy.feature_extract
    actor_net_noise = []
    for p in actor_net.parameters():
        noise = np.random.normal(size=p.data.size())
        noise_t = torch.FloatTensor(noise)
        actor_net_noise.append(noise_t)
    gnn_net_noise = []
    for p in gnn_net.parameters():
        noise = np.random.normal(size=p.data.size())
        noise_t = torch.FloatTensor(noise)
        gnn_net_noise.append(noise_t)
    return actor_net_noise, gnn_net_noise

def sign(k_id): return -1. if k_id % 2 == 0 else 1.  # mirrored sampling

def rollout(env, agent, seed_and_id=None,):
    # perturb parameters using seed
    actor_net = agent.policy.actor
    gnn_net = agent.policy.feature_extract
    old_params_actor = actor_net.state_dict()
    old_params_gnn = gnn_net.state_dict()
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        np.random.seed(seed)
        actor_net_noise, gnn_net_noise = sample_noise(agent)
        for p, p_n in zip(actor_net.parameters(), actor_net_noise):
            p.data += NOISE_STD * p_n * sign(k_id)
        for p, p_n in zip(gnn_net.parameters(), gnn_net_noise):
            p.data += NOISE_STD * p_n * sign(k_id)

    n_m = random.randint(5, 9)
    n_j = random.randint(n_m, 9)
    data_generator = uni_instance_gen
    proctime_matrix, m_matrix = data_generator(n_j=n_j, n_m=n_m, low=configs.low, high=configs.high)
    fea, adj, _, reward, candidate, mask,done  = env.reset(machine_matrix=m_matrix, processing_time_matrix=proctime_matrix)
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,  # graph_pool_type=average
                             batch_size=torch.Size([1, n_j * n_m, n_j * n_m]),
                             n_nodes=n_j * n_m,
                             device=device)
    ep_r = 0
    while True:
        fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
        adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        mask_tensor = torch.from_numpy(np.copy(mask)).to(device)

        with torch.no_grad():
            pi, _ = agent.policy(x=fea_tensor,
                                 graph_pool=g_pool_step,
                                 n_j=n_j,
                                 padded_nei=None,
                                 adj=adj_tensor,
                                 candidate=candidate_tensor.unsqueeze(0),
                                 mask=mask_tensor.unsqueeze(0))
            action = greedy_select_action(pi, candidate)
            fea, adj, _, reward, candidate, mask, done = env.step(action)

        if done:
            makespan = env.global_time
            ep_r = -(makespan/np.sum(proctime_matrix))
            break
    actor_net.load_state_dict(old_params_actor)
    gnn_net.load_state_dict(old_params_gnn)
    return ep_r

def ES_train(agent, utility, env):
    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID, dtype=np.uint32).repeat(2)  # mirrored sampling
    # distribute training in parallel
    jobs = [
        pool.apply_async(rollout, (env, agent, [noise_seed[k_id], k_id],)) for k_id in range(N_KID * 2)]
    rewards = np.array([j.get() for j in jobs])
    rewards -= np.mean(rewards)
    s = np.std(rewards)
    if abs(s) > 1e-6:
        rewards /= s
    # kids_rank = np.argsort(rewards)[::-1]  # rank kid id by reward
    actor_weighted_noise = None
    gnn_weighted_noise = None
    # for ui, k_id in enumerate(kids_rank):
    #     np.random.seed(noise_seed[k_id])
    #     actor_net_noise, gnn_net_noise = sample_noise(agent)
    #     if actor_weighted_noise is None:
    #         actor_weighted_noise = [utility[ui] * sign(k_id) * p_n for p_n in actor_net_noise]
    #         gnn_weighted_noise = [utility[ui] * sign(k_id) * p_n for p_n in gnn_net_noise]
    #     else:
    #         for w_n, p_n in zip(actor_weighted_noise, actor_net_noise):
    #             w_n += utility[ui] * sign(k_id) * p_n
    #         for w_n, p_n in zip(gnn_weighted_noise, gnn_net_noise):
    #             w_n += utility[ui] * sign(k_id) * p_n
    for ui, _ in enumerate(noise_seed):
        np.random.seed(noise_seed[ui])
        actor_net_noise, gnn_net_noise = sample_noise(agent)
        if actor_weighted_noise is None:
            actor_weighted_noise = [rewards[ui] * sign(ui) * p_n for p_n in actor_net_noise]
            gnn_weighted_noise = [rewards[ui] * sign(ui) * p_n for p_n in gnn_net_noise]
        else:
            for w_n, p_n in zip(actor_weighted_noise, actor_net_noise):
                w_n += rewards[ui] * sign(ui) * p_n
            for w_n, p_n in zip(gnn_weighted_noise, gnn_net_noise):
                w_n += rewards[ui] * sign(ui) * p_n

    for p, p_update in zip(agent.policy.actor.parameters(), actor_weighted_noise):
        update = p_update / (len(rewards) * NOISE_STD)
        p.data += LEARNING_RATE * update
    for p, p_update in zip(agent.policy.feature_extract.parameters(), gnn_weighted_noise):
        update = p_update / (len(rewards) * NOISE_STD)
        p.data += LEARNING_RATE * update

    # weighted_noise = None
    # norm_reward = np.array(batch_reward)
    # norm_reward -= np.mean(norm_reward)
    # s = np.std(norm_reward)
    # if abs(s) > 1e-6:
    #     norm_reward /= s
    #
    # for noise, reward in zip(batch_noise, norm_reward):
    #     if weighted_noise is None:
    #         weighted_noise = [reward * p_n for p_n in noise]
    #     else:
    #         for w_n, p_n in zip(weighted_noise, noise):
    #             w_n += reward * p_n
    # m_updates = []
    # for p, p_update in zip(net.parameters(), weighted_noise):
    #     update = p_update / (len(batch_reward) * NOISE_STD)
    #     p.data += LR * update
    #     m_updates.append(torch.norm(update))

if __name__ == '__main__':
    # set tensorboard
    tensorboard_enable = input("Whether the tensorboard is enabled:")
    TIMESTAMP = time.strftime("%m-%d-%H-%M", time.localtime(time.time()))
    if bool(tensorboard_enable):
        comment = "ES training after ppo train"
        writer = SummaryWriter(
            log_dir=f'runs_multiInstance/' + TIMESTAMP + comment)
        print(tensorboard_enable, ':', TIMESTAMP + comment)

    # load the net
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
    # path = './SavedNetwork/{}.pth'.format("mixed_training_1_99_01-27-22-59")
    # ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    # utility instead reward for update parameters (rank transformation)

    base = N_KID * 2  # *2 for mirrored sampling
    rank = np.arange(1, base + 1)
    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base

    # create envs and validate instance data
    env = JSSPSimulator(num_jobs=None, num_machines=None)
    data_generator = uni_instance_gen
    np.random.seed(configs.np_seed_train)
    random.seed(configs.python_seed)
    torch.manual_seed(configs.torch_seed)

    vali_data3 = []
    for i in range(100):
        n_m = np.random.randint(5,9)
        n_j = np.random.randint(n_m,9)
        proctime_matrix,m_matrix= data_generator(n_j=n_j, n_m=n_m, low=configs.low, high=configs.high)
        vali_data3.append((proctime_matrix,m_matrix))

    pool = mp.Pool(processes=N_CORE)

    # test ppo trained network before ES_training
    vali_result = validate(vali_data3, ppo.policy).mean()
    print('The validation quality is:', vali_result)
    # for item in vali_result:
    #     print(item)

    if tensorboard_enable:
        writer.add_scalar("vali_result", vali_result, 0)
    record = vali_result

    # training
    t0 = time.time()
    for i_update in range(configs.max_updates):  # max_updates = 40000
        np.random.seed(i_update)
        ES_train(ppo, utility, env)
        # if vali_result < record:
        #     torch.save(ppo.policy.state_dict(), './SavedNetwork/{}.pth'.format(
        #         'mixed_training' + '_' + str(configs.low) + '_' + str(configs.high) + "_" + TIMESTAMP))
        #     record = vali_result
        vali_result = validate(vali_data3, ppo.policy).mean()
        print('The validation quality is:', vali_result)
        if tensorboard_enable:
            writer.add_scalar("vali_result", vali_result, i_update)



