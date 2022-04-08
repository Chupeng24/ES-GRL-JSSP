import gym
import time
import numpy as np
from Params import configs
import torch
import torch.nn as nn
from PPO_jssp_multiInstances_update import PPO
from torch.utils.tensorboard import SummaryWriter
from uniform_instance_gen import uni_instance_gen
from pyjssp.simulators import JSSPSimulator
from validation_update import validate
import random
from agent_utils import *
from mb_agg import *


MAX_BATCH_EPISODES = 100
MAX_BATCH_STEPS = 10000
NOISE_STD = 0.05
LEARNING_RATE = 0.01
device = torch.device("cpu")

base = MAX_BATCH_EPISODES * 2  # *2 for mirrored sampling
rank = np.arange(1, base + 1)
util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
utility = util_ / util_.sum() - 1 / base

data_generator = uni_instance_gen
np.random.seed(configs.np_seed_train)
random.seed(configs.python_seed)
torch.manual_seed(configs.torch_seed)

vali_data3 = []
for i in range(100):
    n_m = np.random.randint(5, 9)
    n_j = np.random.randint(n_m, 9)
    proctime_matrix, m_matrix = data_generator(n_j=n_j, n_m=n_m, low=configs.low, high=configs.high)
    vali_data3.append((proctime_matrix, m_matrix))


class Net(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


def evaluate(env, agent):
    # np.random.seed(configs.np_seed_train)
    # random.seed(configs.python_seed)
    n_m = random.randint(5, 9)
    n_j = random.randint(n_m, 9)
    data_generator = uni_instance_gen
    proctime_matrix, m_matrix = data_generator(n_j=n_j, n_m=n_m, low=configs.low, high=configs.high)
    # proctime_matrix, m_matrix = vali_data3[0]
    fea, adj, _, reward, candidate, mask, done = env.reset(machine_matrix=m_matrix, processing_time_matrix=proctime_matrix)
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,  # graph_pool_type=average
                             batch_size=torch.Size([1, n_j * n_m, n_j * n_m]),
                             n_nodes=n_j * n_m,
                             device=device)
    ep_r = 0
    steps = 0
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
            ep_r = -(makespan / np.sum(proctime_matrix))
            # ep_r = - float(makespan)
            break
    return ep_r, steps


def sample_noise(agent):
    actor_net = agent.policy.actor
    gnn_net = agent.policy.feature_extract
    actor_net_pos = []
    actor_net_neg = []
    for p in actor_net.parameters():
        noise = np.random.normal(size=p.data.size())
        noise_t = torch.FloatTensor(noise)
        actor_net_pos.append(noise_t)
        actor_net_neg.append(-noise_t)
    gnn_net_pos = []
    gnn_net_neg = []
    for p in gnn_net.parameters():
        noise = np.random.normal(size=p.data.size())
        noise_t = torch.FloatTensor(noise)
        gnn_net_pos.append(noise_t)
        gnn_net_neg.append(-noise_t)
    return actor_net_pos, actor_net_neg, gnn_net_pos, gnn_net_neg


def eval_with_noise(env, agent, actor_net_noise, gnn_net_noise):
    actor_net = agent.policy.actor
    gnn_net = agent.policy.feature_extract
    old_params_actor = actor_net.state_dict()
    old_params_gnn = gnn_net.state_dict()
    for p, p_n in zip(actor_net.parameters(), actor_net_noise):
        p.data += NOISE_STD * p_n
    for p, p_n in zip(gnn_net.parameters(), gnn_net_noise):
        p.data += NOISE_STD * p_n
    r, s = evaluate(env, agent)
    actor_net.load_state_dict(old_params_actor)
    gnn_net.load_state_dict(old_params_gnn)
    return r, s


def train_step(agent, actor_batch_noise, gnn_batch_noise, batch_reward, writer, step_idx):
    actor_net = agent.policy.actor
    gnn_net = agent.policy.feature_extract
    weighted_noise = None
    norm_reward = np.array(batch_reward)
    norm_reward -= np.mean(norm_reward)
    s = np.std(norm_reward)
    if abs(s) > 1e-6:
        norm_reward /= s

    kids_rank = np.argsort(norm_reward)[::-1]

    for idx, noise in enumerate(actor_batch_noise):
        util_value = utility[np.where(kids_rank == idx)][0]
        # util_value = torch.from_numpy(util_value)
        if weighted_noise is None:
            weighted_noise = [util_value * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += util_value * p_n
    # for noise, reward in zip(actor_batch_noise, norm_reward):
    #     if weighted_noise is None:
    #         weighted_noise = [reward * p_n for p_n in noise]
    #     else:
    #         for w_n, p_n in zip(weighted_noise, noise):
    #             w_n += reward * p_n
    #cm_updates = []
    for p, p_update in zip(actor_net.parameters(), weighted_noise):
        update = p_update / (len(batch_reward) * NOISE_STD)
        # p.data += LEARNING_RATE * update
        p.data = 0.9 * p.data +  0.1 * LEARNING_RATE * update
    weighted_noise = None
    # for noise, reward in zip(gnn_batch_noise, norm_reward):
    #     if weighted_noise is None:
    #         weighted_noise = [reward * p_n for p_n in noise]
    #     else:
    #         for w_n, p_n in zip(weighted_noise, noise):
    #             w_n += reward * p_n
    for idx, noise in enumerate(gnn_batch_noise):
        util_value = utility[np.where(kids_rank == idx)][0]
        # util_value = torch.from_numpy(util_value)
        if weighted_noise is None:
            weighted_noise = [util_value * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += util_value * p_n
    for p, p_update in zip(gnn_net.parameters(), weighted_noise):
        update = p_update / (len(batch_reward) * NOISE_STD)
        # p.data += LEARNING_RATE * update
        p.data = 0.9 * p.data +  0.1 * LEARNING_RATE * update

if __name__ == "__main__":
    # set tensorboard
    tensorboard_enable = input("Whether the tensorboard is enabled:")
    TIMESTAMP = time.strftime("%m-%d-%H-%M", time.localtime(time.time()))
    if bool(tensorboard_enable):
        comment = "single ES training after ppo train"
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
    path = './SavedNetwork/{}.pth'.format("mixed_training_1_99_01-27-22-59")
    # ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    # create envs and validate instance data
    env = JSSPSimulator(num_jobs=None, num_machines=None)
    # test ppo trained network before ES_training
    vali_result = validate(vali_data3, ppo.policy).mean()
    print(vali_result)
    # for item in vali_result:
    #     print(item)
    if tensorboard_enable:
        writer.add_scalar("vali_result", vali_result, 0)
    record = vali_result
    # net = Net(env.observation_space.shape[0], env.action_space.n)
    # print(net)
    for i_update in range(configs.max_updates):
        t_start = time.time()
        actor_batch_noise = []
        gnn_batch_noise = []
        batch_reward = []
        batch_steps = 0
        for _ in range(MAX_BATCH_EPISODES):
            actor_net_pos, actor_net_neg, gnn_net_pos, gnn_net_neg = sample_noise(ppo)
            actor_batch_noise.append(actor_net_pos)
            actor_batch_noise.append(actor_net_neg)
            gnn_batch_noise.append(gnn_net_pos)
            gnn_batch_noise.append(gnn_net_neg)
            reward, steps = eval_with_noise(env, ppo, actor_net_pos, gnn_net_pos)
            batch_reward.append(reward)
            batch_steps += steps
            reward, steps = eval_with_noise(env, ppo, actor_net_neg, gnn_net_neg)
            batch_reward.append(reward)
            batch_steps += steps

        m_reward = np.mean(batch_reward)

        train_step(ppo, actor_batch_noise, gnn_batch_noise, batch_reward,
                   writer, i_update)
        vali_result = validate(vali_data3, ppo.policy).mean()
        print(vali_result)
        # for item in vali_result:
        #     print(item)
        if tensorboard_enable:
            writer.add_scalar("vali_result", vali_result, i_update + 1)
        # writer.add_scalar("reward_mean", m_reward, i_update)
        # writer.add_scalar("reward_std", np.std(batch_reward),
        #                   i_update)
        # writer.add_scalar("reward_max", np.max(batch_reward),
        #                   i_update)
        # writer.add_scalar("batch_episodes", len(batch_reward),
        #                   i_update)
        # writer.add_scalar("batch_steps", batch_steps, i_update)

    pass