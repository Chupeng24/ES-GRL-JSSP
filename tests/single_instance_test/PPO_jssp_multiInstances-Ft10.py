from mb_agg import *
from tests.agent_utils import eval_actions
from tests.agent_utils import select_action
from models.actor_critic import ActorCritic
from copy import deepcopy
import torch
import time
import torch.nn as nn
import numpy as np
from Params import configs
from tensorboardX import SummaryWriter
from pyjssp.benchmarks import FT10

TIMESTAMP= time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))
writer = SummaryWriter(log_dir='runs-Ft10/'+ TIMESTAMP)
device = torch.device(configs.device)

class Memory:
    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        self.candidate_mb = []
        self.mask_mb = []
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.logprobs = []

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.candidate_mb[:]
        del self.mask_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.logprobs[:]


class PPO:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 n_j,
                 n_m,
                 num_layers,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(n_j=n_j,
                                  n_m=n_m,
                                  num_layers=num_layers,
                                  learn_eps=False,
                                  neighbor_pooling_type=neighbor_pooling_type,
                                  input_dim=input_dim,
                                  hidden_dim=hidden_dim,
                                  num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                                  num_mlp_layers_actor=num_mlp_layers_actor,
                                  hidden_dim_actor=hidden_dim_actor,
                                  num_mlp_layers_critic=num_mlp_layers_critic,
                                  hidden_dim_critic=hidden_dim_critic,
                                  device=device)
        self.policy_old = deepcopy(self.policy)

        '''self.policy.load_state_dict(
            torch.load(path='./{}.pth'.format(str(n_j) + '_' + str(n_m) + '_' + str(1) + '_' + str(99))))'''

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)  # 每隔2000epoch，lr*0.9 https://blog.csdn.net/qyhaill/article/details/103043637

        self.V_loss_2 = nn.MSELoss()

    def update(self, memory, n_tasks, g_pool):

        vloss_coef = configs.vloss_coef          # coefficient 系数
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef

        # rewards_all_env = []
        # adj_mb_t_all_env = []
        # fea_mb_t_all_env = []
        # candidate_mb_t_all_env = []
        # mask_mb_t_all_env = []
        # a_mb_t_all_env = []
        # old_logprobs_mb_t_all_env = []
        # store data for all env
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.r_mb), reversed(memory.done_mb)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)     # reward 还作为一个归一化
        # process each env data
        adj_mb= aggr_obs(torch.stack(memory.adj_mb).to(device), n_tasks)
        fea_mb = torch.stack(memory.fea_mb).to(device)
        fea_mb = fea_mb.reshape(-1, fea_mb.size(-1))
        candidate_mb = torch.stack(memory.candidate_mb).to(device)
        if len(memory.candidate_mb) == 1 :
            candidate_mb.unsqueeze(0)
        mask_mb = torch.stack(memory.mask_mb).to(device).squeeze()
        a_mb = torch.stack(memory.a_mb).to(device).squeeze()
        old_logprobs_mb = torch.stack(memory.logprobs).to(device).squeeze().detach()

        # get batch argument for net forwarding: mb_g_pool is same for all env
        mb_g_pool = g_pool_cal(g_pool, torch.stack(memory.adj_mb).to(device).shape, n_tasks, device)

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            loss_sum = 0
            vloss_sum = 0

            pis, vals = self.policy(x=fea_mb,
                                    graph_pool=mb_g_pool,
                                    adj=adj_mb,
                                    candidate=candidate_mb,
                                    mask=mask_mb,
                                    padded_nei=None)
            logprobs, ent_loss = eval_actions(pis.squeeze(), a_mb)
            ratios = torch.exp(logprobs - old_logprobs_mb.detach())
            advantages = rewards - vals.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            v_loss = self.V_loss_2(vals.squeeze(), rewards)
            p_loss = - torch.min(surr1, surr2)
            ent_loss = - ent_loss.clone()
            loss = vloss_coef * v_loss + ploss_coef * p_loss + entloss_coef * ent_loss
            loss_sum += loss
            vloss_sum += v_loss
            self.optimizer.zero_grad()
            loss_sum.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        if configs.decayflag:
            self.scheduler.step()
        return loss_sum.mean().item(), vloss_sum.mean().item(), -ent_loss.mean().item()


def main():

    from pyjssp.simulators import JSSPSimulator
    env = JSSPSimulator(num_jobs=configs.n_j, num_machines=configs.n_m) # 初始化环境
    env = FT10()
    # from uniform_instance_gen import uni_instance_gen
    # data_generator = uni_instance_gen
    # # n_j=15, n_m=15, np_seed_validation=200
    # dataLoaded = np.load('./DataGen/generatedData' + str(configs.n_j) + '_' + str(configs.n_m) + '_Seed' + str(configs.np_seed_validation) + '.npy')
    vali_data = []
    # for i in range(dataLoaded.shape[0]):
    #     vali_data.append((dataLoaded[i][0], dataLoaded[i][1]))
    vali_data.append((env.processing_time_matrix,env.machine_matrix))
    # torch_seed=600
    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    np.random.seed(configs.np_seed_train)   # np_seed_train = 200
    memory = Memory()

    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=configs.n_j,   # n_j=15, n_m=15
              n_m=configs.n_m,
              num_layers=configs.num_layers,    # num_layers=3
              neighbor_pooling_type=configs.neighbor_pooling_type,   # neighbor_pooling_type=sum
              input_dim=configs.input_dim,                           # input_dim=2
              hidden_dim=configs.hidden_dim,                         # hidden_dim=64
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract, #=2
              num_mlp_layers_actor=configs.num_mlp_layers_actor,                     #=2
              hidden_dim_actor=configs.hidden_dim_actor,                             #=32
              num_mlp_layers_critic=configs.num_mlp_layers_critic,                   #=2
              hidden_dim_critic=configs.hidden_dim_critic)                           #=32
    print(configs.neighbor_pooling_type) # neighbor_pooling_type=sum
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,  # graph_pool_type=average
                             batch_size=torch.Size([1, configs.n_j*configs.n_m, configs.n_j*configs.n_m]),
                             n_nodes=configs.n_j*configs.n_m,
                             device=device)
    # training loop
    log = []
    validation_log = []
    optimal_gaps = []
    optimal_gap = 1
    record = 100000
    for i_update in range(configs.max_updates):      # max_updates = 40000

        t3 = time.time()

        ep_reward = 0
        # adj = []
        # fea = []
        # candidate = []
        # mask = []
        ep_makespan = 0

        #proctime_matrix,m_matrix= data_generator(n_j=configs.n_j, n_m=configs.n_m, low=configs.low, high=configs.high)
        m_matrix = env.machine_matrix
        proctime_matrix = env.processing_time_matrix
        fea, adj, _, reward, candidate, mask,done  = env.reset(machine_matrix=m_matrix, processing_time_matrix=proctime_matrix)

        # rollout the env
        while True:
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
            adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device)

            with torch.no_grad():
                pi, _ = ppo.policy_old(x=fea_tensor,
                                       graph_pool=g_pool_step,
                                       padded_nei=None,
                                       adj=adj_tensor,
                                       candidate=candidate_tensor.unsqueeze(0),
                                       mask=mask_tensor.unsqueeze(0))
                try:
                    action, a_idx = select_action(pi, candidate, memory)
                except ValueError as e:
                    print(pi)
                    print("#############################################")
            # adj_envs = []
            # fea_envs = []
            # candidate_envs = []
            # mask_envs = []
            # Saving episode data
            memory.adj_mb.append(adj_tensor)
            memory.fea_mb.append(fea_tensor)
            memory.candidate_mb.append(candidate_tensor)
            memory.mask_mb.append(mask_tensor)
            memory.a_mb.append(a_idx)

            fea, adj, _, reward, candidate, mask,done = env.step(action)
            # adj_envs.append(adj)
            # fea_envs.append(fea)
            # candidate_envs.append(candidate)
            # mask_envs.append(mask)
            ep_reward += reward
            memory.r_mb.append(reward)
            memory.done_mb.append(done)
            if done:
                ep_makespan = env.global_time
                break
        # for j in range(configs.num_envs):
        #     ep_rewards[j] -= envs[j].posRewards

        loss, v_loss, ent_loss = ppo.update(memory, configs.n_j*configs.n_m, configs.graph_pool_type)
        memory.clear_memory()
        # log.append([i_update, ep_reward])
        if i_update % 100 == 0:
            # file_writing_obj = open('./' + 'log_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            # file_writing_obj.write(str(log))
            writer.add_scalar("Train loss", loss, i_update)  #
            writer.add_scalar("Train reward", ep_reward, i_update)  #
            writer.add_scalar("policy_entropy", ent_loss, i_update)
            writer.add_scalar("makespan", ep_makespan, i_update)

        # log results
        print('Episode {}\t Last reward: {:.2f}\t makespan: {}\t Mean_Vloss: {:.8f}'.format(
            i_update, ep_reward, ep_makespan, v_loss))
        
        # validate and save use mean performance
        # t4 = time.time()
        # if i_update!=0 and i_update % 99 == 0:
        #     print("####################################")
        #     vali_result = validate(vali_data, ppo.policy)
        #     # validation_log.append(vali_result)
        #     # if vali_result < record:
        #     #     torch.save(ppo.policy.state_dict(), './{}.pth'.format(
        #     #         str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high)))
        #     #     record = vali_result
        #     print('The validation quality is:', vali_result)
        #     # file_writing_obj1 = open(
        #     #     './' + 'vali_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
        #     # file_writing_obj1.write(str(validation_log))
        #     writer.add_scalar("vali_result", vali_result, i_update)  #
        # t5 = time.time()
    writer.close()

        # print('Training:', t4 - t3)
        # print('Validation:', t5 - t4)


if __name__ == '__main__':
    total1 = time.time()
    main()
    total2 = time.time()
    # print(total2 - total1)