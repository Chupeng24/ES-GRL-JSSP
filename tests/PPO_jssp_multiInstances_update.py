import random
from mb_agg import *
from agent_utils import eval_actions
from agent_utils import select_action
from models.actor_critic import ActorCritic
from copy import deepcopy
import torch
import time
import torch.nn as nn
import numpy as np
from Params import configs
from validation_update import validate
from torch.utils.tensorboard import SummaryWriter

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
                                  learn_eps=True,
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

    def update(self, memories, n_j,n_m, g_pool):

        vloss_coef = configs.vloss_coef          # coefficient 系数
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef

        rewards_all_env = []
        adj_mb_t_all_env = []
        fea_mb_t_all_env = []
        candidate_mb_t_all_env = []
        mask_mb_t_all_env = []
        a_mb_t_all_env = []
        old_logprobs_mb_t_all_env = []
        mb_g_pool = []
        # store data for all env
        for i in range(len(memories)):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memories[i].r_mb), reversed(memories[i].done_mb)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            if rewards.shape[0] > 1:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
                #rewards = rewards / (rewards.std() + 1e-5)
            else:
                pass
            rewards_all_env.append(rewards)
            # process each env data
            adj_mb_t_all_env.append(aggr_obs(torch.stack(memories[i].adj_mb).to(device), n_j[i] * n_m[i]))
            fea_mb_t = torch.stack(memories[i].fea_mb).to(device)
            fea_mb_t = fea_mb_t.reshape(-1, fea_mb_t.size(-1))
            fea_mb_t_all_env.append(fea_mb_t)
            if len(memories[i].candidate_mb) == 1 :
                candidate_mb_t_all_env.append(torch.stack(memories[i].candidate_mb).to(device))
            else:
                candidate_mb_t_all_env.append(torch.stack(memories[i].candidate_mb).to(device).squeeze())
            mask_mb_t_all_env.append(torch.stack(memories[i].mask_mb).to(device).squeeze())
            a_mb_t_all_env.append(torch.stack(memories[i].a_mb).to(device).squeeze())
            old_logprobs_mb_t_all_env.append(torch.stack(memories[i].logprobs).to(device).squeeze().detach())

            # get batch argument for net forwarding: mb_g_pool is same for all env
            mb_g_pool.append(g_pool_cal(g_pool, torch.stack(memories[i].adj_mb).to(device).shape, n_j[i] * n_m[i], device))

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            loss_sum = 0
            vloss_sum = 0
            entloss_sum = 0

            for i in range(len(memories)):
                pis, vals = self.policy(x=fea_mb_t_all_env[i],
                                        graph_pool=mb_g_pool[i],
                                        n_j = n_j[i],
                                        adj=adj_mb_t_all_env[i],
                                        candidate=candidate_mb_t_all_env[i],
                                        mask=mask_mb_t_all_env[i],
                                        padded_nei=None)
                logprobs, ent_loss = eval_actions(pis.squeeze(), a_mb_t_all_env[i])
                ratios = torch.exp(logprobs - old_logprobs_mb_t_all_env[i].detach())
                advantages = rewards_all_env[i] - vals.detach().view(-1)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                v_loss = self.V_loss_2(vals.view(-1), rewards_all_env[i])
                p_loss = - torch.min(surr1, surr2).mean()
                ent_loss = - ent_loss.clone()
                loss = vloss_coef * v_loss + ploss_coef * p_loss + entloss_coef * ent_loss
                loss_sum += loss
                vloss_sum += v_loss
                entloss_sum += ent_loss
            self.optimizer.zero_grad()
            loss_sum.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        if configs.decayflag:
            self.scheduler.step()
        return loss_sum.mean().item(), vloss_sum.mean().item(),-entloss_sum.mean().item()


def main():

    from pyjssp.simulators import JSSPSimulator
    env = JSSPSimulator(num_jobs=None, num_machines=None)

    from uniform_instance_gen import uni_instance_gen
    data_generator = uni_instance_gen

    tensorboard_enable = input("Whether the tensorboard is enabled:")
    TIMESTAMP = time.strftime("%m-%d-%H-%M", time.localtime(time.time()))
    if bool(tensorboard_enable):
        comment = input("what changes have been made:")
        writer = SummaryWriter(log_dir=f'runs_multiInstance/'+ TIMESTAMP +'-mt'+f'-lr-{configs.lr}'+f'-entcoef-{configs.entloss_coef}-'+comment)
        print(tensorboard_enable,':',TIMESTAMP+ '-mixed_training')

    # n_j=15, n_m=15, np_seed_validation=200
    flag_temp = False
    if flag_temp:
        dataLoaded1 = np.load('./DataGen/generatedData' + str(6) + '_' + str(6) + '_Seed' + str(configs.np_seed_validation) + '.npy')
        vali_data1 = []
        for i in range(dataLoaded1.shape[0]):
            vali_data1.append((dataLoaded1[i][0], dataLoaded1[i][1]))

        dataLoaded2 = np.load('./DataGen/generatedData' + str(10) + '_' + str(10) + '_Seed' + str(configs.np_seed_validation) + '.npy')
        vali_data2 = []
        for i in range(dataLoaded1.shape[0]):
            vali_data2.append((dataLoaded2[i][0], dataLoaded2[i][1]))

    vali_data3 = []
    np.random.seed(configs.np_seed_train)
    random.seed(configs.python_seed)
    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        print("using GPU")
        torch.cuda.manual_seed_all(configs.torch_seed) # torch_seed=600

    for i in range(100):
        n_m = np.random.randint(5,9)
        n_j = np.random.randint(n_m,9)
        proctime_matrix,m_matrix= data_generator(n_j=n_j, n_m=n_m, low=configs.low, high=configs.high)
        vali_data3.append((proctime_matrix,m_matrix))
    np.random.seed(configs.np_seed_train)
    memories = [Memory() for _ in range(configs.num_envs)]

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

    # training loop
    log = []
    validation_log = []
    optimal_gaps = []
    optimal_gap = 1
    record = 100000
    # vali_result = validate(vali_data, ppo.policy).mean()
    # writer.add_scalar("vali_result", vali_result, 0)  #
    # print('The validation quality is:', vali_result)
    for i_update in range(configs.max_updates):      # max_updates = 40000

        t3 = time.time()

        ep_rewards = []
        # adj = []
        # fea = []
        # candidate = []
        # mask = []
        ep_makespan = []
        n_j_list = []
        n_m_list = []
        for i in range(configs.num_envs):
            ep_reward = 0
            n_m = random.randint(5,9)
            n_j = random.randint(n_m,9)
            n_m_list.append(n_m)
            n_j_list.append(n_j)
            proctime_matrix,m_matrix= data_generator(n_j=n_j, n_m=n_m, low=configs.low, high=configs.high)
            np.random.seed(i_update*10+i)
            # fea, adj, _, reward, candidate, mask,done  = env.reset(machine_matrix=m_matrix, processing_time_matrix=proctime_matrix,proctime_std=2,sched_ratio=0.3,mbrk_Ag=0.05,mbrk_seed=i_update*10+i+1)
            fea, adj, _, reward, candidate, mask,done  = env.reset(machine_matrix=m_matrix, processing_time_matrix=proctime_matrix)
            g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,  # graph_pool_type=average
                                     batch_size=torch.Size([1, n_j*n_m, n_j*n_m]),
                                     n_nodes=n_j*n_m,
                                     device=device)
            # rollout the env
            while True:
                fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
                adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
                candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
                mask_tensor = torch.from_numpy(np.copy(mask)).to(device)

                with torch.no_grad():
                    pi, _ = ppo.policy_old(x=fea_tensor,
                                           graph_pool=g_pool_step,
                                           n_j = n_j,
                                           padded_nei=None,
                                           adj=adj_tensor,
                                           candidate=candidate_tensor.unsqueeze(0),
                                           mask=mask_tensor.unsqueeze(0))
                    # if torch.any(torch.isnan(next(ppo.policy_old.feature_extract.mlps[0].linears[0].named_parameters()))):
                    #     print("error")
                    action, a_idx = select_action(pi, candidate, memories[i])
                # adj_envs = []
                # fea_envs = []
                # candidate_envs = []
                # mask_envs = []
                # Saving episode data
                memories[i].adj_mb.append(adj_tensor)
                memories[i].fea_mb.append(fea_tensor)
                memories[i].candidate_mb.append(candidate_tensor)
                memories[i].mask_mb.append(mask_tensor)
                memories[i].a_mb.append(a_idx)

                fea, adj, _, reward, candidate, mask,done = env.step(action)
                # adj_envs.append(adj)
                # fea_envs.append(fea)
                # candidate_envs.append(candidate)
                # mask_envs.append(mask)
                ep_reward += reward
                memories[i].r_mb.append(reward)
                memories[i].done_mb.append(done)
                if done:
                    ep_makespan.append(env.global_time)
                    ep_rewards.append(ep_reward)
                    break
        # for j in range(configs.num_envs):
        #     ep_rewards[j] -= envs[j].posRewards
        loss, v_loss, ent_loss = ppo.update(memories, n_j_list,n_m_list, configs.graph_pool_type)
        n_j_list.clear()
        n_m_list.clear()
        for memory in memories:
            memory.clear_memory()
        log.append([i_update, np.mean(ep_rewards)])
        if i_update % 50 == 0:
            file_writing_obj = open('./log/' + 'log_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            file_writing_obj.write(str(log))
            if tensorboard_enable:
                writer.add_scalar("Train loss", loss, i_update)  #
                writer.add_scalar("Train reward", np.mean(ep_rewards), i_update)  #
                writer.add_scalar("policy_entropy", ent_loss, i_update)

        # log results
        print('Episode {}\t Last reward: {:.2f}\t makespan: {}\t Mean_Vloss: {:.8f}'.format(
            i_update, np.mean(ep_rewards), np.mean(ep_makespan), v_loss))

        # validate and save use mean performance
        # t4 = time.time()
        if i_update % 100 == 0:
            print("###############################################################################")
            #vali_result1 = validate(vali_data1, ppo.policy).mean()
            #vali_result2 = validate(vali_data2, ppo.policy).mean()
            vali_result = validate(vali_data3, ppo.policy).mean()
            #vali_result = vali_result1 + vali_result2
            validation_log.append(vali_result)
            if vali_result < record:
                torch.save(ppo.policy.state_dict(), './SavedNetwork/{}.pth'.format('mixed_training' + '_' + str(configs.low) + '_' + str(configs.high) + "_"+ TIMESTAMP))
                record = vali_result
            print('The validation quality is:', vali_result)
            # file_writing_obj1 = open(
            #     './log/' + 'vali_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            # file_writing_obj1.write(str(validation_log))
            if tensorboard_enable:
                writer.add_scalar("vali_result", vali_result, i_update)  #
        # t5 = time.time()
    if tensorboard_enable:
        writer.close()
    # print('Training:', t4 - t3)
    # print('Validation:', t5 - t4)

if __name__ == '__main__':
    total1 = time.time()
    main()
    total2 = time.time()
    # print(total2 - total1)