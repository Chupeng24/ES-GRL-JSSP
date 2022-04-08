import random
from collections import OrderedDict
import math
import gym
import numpy as np
import rope.base.resources
from gym.utils import EzPickle
import matplotlib.pyplot as plt
import networkx as nx

from pyjssp.jobShopSamplers import jssp_sampling
from pyjssp.operationHelpers import (JobManager,
                                     NodeProcessingTimeJobManager,
                                     get_edge_color_map,
                                     get_node_color_map)
from pyjssp.machineHelpers import (MachineManager,
                                   NodeProcessingTimeMachineManager)
from pyjssp.configs import (N_SEP, SEP, NEW)
from pyjssp.configs import (NOT_START_NODE_SIG,
                            PROCESSING_NODE_SIG,
                            DONE_NODE_SIG)
from pyjssp.updateEntTimeLB import calEndTimeLB
from pyjssp.gen_Adj import gen_adj_matrix
from pyjssp.dispatchRules import dispatch_rule_func

class JSSPSimulator(gym.Env, EzPickle):
    def __init__(self,
                 num_machines=None,
                 num_jobs=None,
                 detach_done=False,
                 name=None,
                 use_surrogate_index=True,
                 # delay=False,
                 verbose=False):
        EzPickle.__init__(self)

        # OK
        if name is None:
            self.name = '{} machine {} job'.format(num_machines, num_jobs)
        else:
            self.name = name

        self.num_machine = num_machines
        self.detach_done = detach_done
        self.num_jobs = num_jobs
        self.num_steps = num_machines
        self.use_surrogate_index = use_surrogate_index
        # self.delay = delay
        self.verbose = verbose

        #self.num_ops = self.num_jobs * self.num_steps
        #self.Adj = gen_adj_matrix(self, self.num_ops, self.num_steps)
        self.et_normalize_coef = 1000
        self.high = 99
        # simulation procedure : global_time +=1 -> do_processing -> transit
    # OK
    def reset(self,
              processing_time_matrix=None,
              machine_matrix=None,
              jssp_path=None,
              sched_ratio=None,
              proctime_std= 0,
              mbrk_Ag = 0,
              mbrk_seed=None,
              proc_seed=None,
              sched_seed=None):
        if machine_matrix is None or processing_time_matrix is None and jssp_path is not None:
            ms,prts = self.from_path(jssp_path)
            self.machine_matrix = ms.astype(int)
            self.processing_time_matrix = prts.astype(float)
        elif machine_matrix is None or processing_time_matrix is None:
            ms, prts = self._sample_jssp_graph(self.num_machine, self.num_jobs)
            self.machine_matrix = ms.astype(int)
            self.processing_time_matrix = prts.astype(float)
        else:
            self.machine_matrix = machine_matrix.astype(int)
            self.processing_time_matrix = processing_time_matrix.astype(float)
        self.num_jobs, self.num_machine = self.machine_matrix.shape
        self.num_steps = self.num_machine
        self.num_ops = self.num_jobs * self.num_machine
        self.scheduled_op = 0
        self.sched_ratio = sched_ratio
        self.random_stop_flag = True

        self.proctime_std = proctime_std
        if proctime_std:
            np.random.seed(proc_seed)
            random.seed(proc_seed+10)
            self.prac_proc_time_matrix = np.random.normal(loc = self.processing_time_matrix,scale= self.proctime_std)
            for job_id,job in enumerate(self.prac_proc_time_matrix):
                for step_id,_ in enumerate(job):
                    # if self.prac_proc_time_matrix[job_id][step_id] > self.processing_time_matrix[job_id][step_id]:
                    #     self.prac_proc_time_matrix[job_id][step_id] = math.ceil(self.prac_proc_time_matrix[job_id][step_id])
                    # else:
                    #     self.prac_proc_time_matrix[job_id][step_id] = math.floor(self.prac_proc_time_matrix[job_id][step_id])
                    self.prac_proc_time_matrix[job_id][step_id] = round(self.prac_proc_time_matrix[job_id][step_id],0)
                    # if self.prac_proc_time_matrix[job_id][step_id] <= 0:
                    #     self.prac_proc_time_matrix[job_id][step_id] = 1
        else:
            self.prac_proc_time_matrix = None
        self.temp1 = np.zeros_like(self.processing_time_matrix, dtype=np.single)
        self.job_manager = JobManager(self.machine_matrix,
                                      self.processing_time_matrix,
                                      use_surrogate_index=self.use_surrogate_index)

        # machine breakdown setup
        self.mbrk_Ag = mbrk_Ag
        self.brk_rep_time_table = None
        if self.mbrk_Ag and mbrk_seed:
            np.random.seed(mbrk_seed)   # In order to create same machine breakdown situation
            random.seed(mbrk_seed+10)
            MOPT = 50
            MTTR = self.mbrk_Ag* 100 * MOPT
            MTBF = ((1-self.mbrk_Ag)*MTTR)/self.mbrk_Ag
            brk_rep_time_sampling = np.zeros((self.num_machine,20),dtype=np.int)
            for row_idx,row in enumerate(brk_rep_time_sampling):
                for cloumn_idx,_ in enumerate(row):
                    if cloumn_idx % 2 == 0:
                        brk_rep_time_sampling[row_idx][cloumn_idx]=np.random.exponential(MTBF)
                    else:
                        brk_rep_time_sampling[row_idx][cloumn_idx]=np.random.exponential(MTTR)
            self.brk_rep_time_table = np.cumsum(brk_rep_time_sampling, axis=1)
        self.machine_manager = MachineManager(self.machine_matrix,
                                              self.job_manager,
                                              # self.delay,
                                              self.proctime_std,
                                              self.prac_proc_time_matrix,
                                              self.temp1,
                                              self.mbrk_Ag,
                                              self.brk_rep_time_table,
                                              self.verbose)
        self.global_time = 0  # -1 matters a lot
        self.proc_time_cp = np.copy(self.processing_time_matrix).astype(np.single)
        self.LBs = calEndTimeLB(self, self.temp1, self.proc_time_cp)
        self.max_endTime = self.LBs.max()
        self.Adj = gen_adj_matrix(self, self.num_ops, self.num_steps)
        # add FDD/MWKR feature
        # temp2 = np.cumsum(self.processing_time_matrix, axis=1,dtype=np.single)
        # temp3 = np.flip(self.processing_time_matrix, axis=1)
        # temp3 = np.cumsum(temp3,axis=1,dtype=np.single)
        # temp3 = np.flip(temp3,axis=1)
        # self.FDDMWKR = temp2 / temp3
        #
        # self.FDDMWKR = (self.FDDMWKR-np.mean(self.FDDMWKR))/np.std(self.FDDMWKR)

        if self.sched_ratio is not None and self.sched_ratio > 0:
            self.random_stop_flag = False
            self.random_op_sum = self.num_ops * self.sched_ratio
            self.random_op_count = 0
            self.random_action_index = 0
            self.random_action_list = np.full(shape=(self.num_ops,),fill_value=np.nan)
            np.random.seed(sched_seed)  # In order to create same machine breakdown situation
            random.seed(sched_seed + 10)
        # self._machine_set = list(set(self.machine_matrix.flatten().tolist()))

        return self.observe()

    def step(self, action=None,disrule_name=None):
        self.transit(action,disrule_name)
        _, _, _, r_trainsit, _, _, _ = self.observe()
        _, cum_reward, _ = self.flush_trivial_ops()

        fea, adj, _, _, candidate, mask,done = self.observe()
        r = r_trainsit + cum_reward
        return fea, adj, _, r, candidate, mask,done

    def process_one_time(self):
        # short_m = float('inf')
        short_op = float('inf')
        short_m  = float('inf')

        for _, machine in self.machine_manager.machines.items():
            if hasattr(machine, 'trans_interval'):
                # if machine.num_done_ops < len(machine.possible_ops):
                if machine.trans_interval < short_m:
                    short_m = machine.trans_interval

        for _, machine in self.machine_manager.machines.items():
            if machine.current_op != None:
                if machine.normal_flag == True:
                    if machine.remaining_time < short_op:
                        short_op = machine.remaining_time

        # if self.mbrk_Ag:
        #     shor_interval = 1
        # else:
        #     shor_interval = int(short_op)
        # if short_m == float('inf') and short_op == float('inf'):
        #     print("")

        shor_interval = int(min(short_m, short_op))

        # if self.mbrk_Ag is not None and self.mbrk_Ag > 0:
        #     shor_interval = 1

        self.global_time += shor_interval
        self.machine_manager.do_processing(self.global_time, shor_interval)

    def transit(self, action=None,disrule_name=None):
        if action is None and disrule_name is None:
            # Perform random action
            #random.seed(200)
            machine = random.choice(self.machine_manager.get_available_machines())
            op_id = random.choice(machine.doable_ops_id)
            job_id, step_id = self.job_manager.sur_index_dict[op_id]
            operation = self.job_manager[job_id][step_id]
            machine.transit(self.global_time, operation)
            row, col = operation._id
            self.temp1[row, col] = self.global_time + operation.processing_time
            if self.sched_ratio is not None and self.sched_ratio > 0:
                if self.random_op_count >= self.random_op_sum:
                    self.random_stop_flag = True
                else:
                    self.random_op_count += 1
                    self.random_action_list[self.random_action_index] = op_id
                    self.random_action_index += 1
        elif action is None and disrule_name is not None:
            # random.seed(200)
            machine = random.choice(self.machine_manager.get_available_machines())
            operation = dispatch_rule_func(self,machine,disrule_name)
            machine.transit(self.global_time, operation)
            row, col = operation._id
            self.temp1[row, col] = self.global_time + operation.processing_time
        elif action is not None and disrule_name is None:
            if self.use_surrogate_index:
                if action in self.job_manager.sur_index_dict.keys():
                    job_id, step_id = self.job_manager.sur_index_dict[action]
                else:
                    raise RuntimeError("Input action is not valid")
            else:
                job_id, step_id = action

            operation = self.job_manager[job_id][step_id]
            machine_id = operation.machine_id
            machine = self.machine_manager[machine_id]

            machine.transit(self.global_time, operation)
            row, col = operation._id
            self.temp1[row, col] = self.global_time + operation.processing_time
        else:
            raise  RuntimeError("transit action is not valid")
        # if self.sched_ratio is not None:
        #     if self.random_op_count >= self.random_op_sum:
        #         self.random_stop_flag = True
        #     else:
        #         self.random_op_count += 1
        self.scheduled_op += 1

    # flush 冲洗；trivial 不重要的
    def flush_trivial_ops(self, reward='idle_time', gamma=1.0):
        done = False
        cum_reward = 0
        while True:
            m_list = []
            do_op_dict = self.get_doable_ops_in_dict()
            all_machine_work = False if bool(do_op_dict) else True

            if all_machine_work:  # all machines are on processing. keep process!
                self.process_one_time()
            else:  # some of machine has possibly trivial action. the others not.
                # load trivial ops to the machines
                num_ops_counter = 1
                for m_id, op_ids in do_op_dict.items():
                    num_ops = len(op_ids)
                    if num_ops == 1:
                        self.transit(op_ids[0])  # load trivial action
                        _, _, _, r, _, _, _ = self.observe(reward)
                        cum_reward = r + gamma * cum_reward
                    else:
                        m_list.append(m_id)
                        num_ops_counter *= num_ops

                # not-all trivial break the loop
                if num_ops_counter != 1:
                    break
            # if simulation is done
            jobs_done = [job.job_done for _, job in self.job_manager.jobs.items()]
            done = True if np.prod(jobs_done) == 1 else False

            if done:
                break
        return m_list, cum_reward, done

    # return available_machines list, OK
    def get_available_machines(self, shuffle_machine=True):
        return self.machine_manager.get_available_machines(shuffle_machine)

    # 如果没有固定机器，则返回字典，字典key为机器id,value为可加工ops列表；若固定机器，则返回doable_ops的列表，OK
    def get_doable_ops_in_dict(self, machine_id=None, shuffle_machine=True):
        if machine_id is None:
            doable_dict = {}
            if self.get_available_machines():
                for m in self.get_available_machines(shuffle_machine):
                    _id = m.machine_id
                    _ops = m.doable_ops_id
                    doable_dict[_id] = _ops
            ret = doable_dict
        else:
            available_machines = [m.machine_id for m in self.get_available_machines()]
            if machine_id in available_machines:
                ret = self.machine_manager[machine_id].doable_ops_id
            else:
                raise RuntimeWarning("Access to the not available machine {}. Return is None".format(machine_id))
        return ret

    # 返回所有机器doable ops的列表
    def get_doable_ops_in_list(self, machine_id=None, shuffle_machine=True):
        doable_dict = self.get_doable_ops_in_dict(machine_id, shuffle_machine)
        do_ops = np.full(shape=self.num_jobs, fill_value=-1, dtype=np.int64)
        undoable_mask = np.full(shape=self.num_jobs, fill_value=False, dtype=bool)
        for _, v_list in doable_dict.items():
            for v in v_list:
                do_ops[v // self.num_steps] = v
        # candidate = np.empty(self.num_steps,dtype=np.int64)
        # do_ops_temp = do_ops//self.num_ops
        # i = 0
        for index, v in enumerate(do_ops):
            if v == -1:
                do_ops[index] = 0
                undoable_mask[index] = True
        #         i += 1
        # if i == self.num_jobs:
        #     print("error")
        return do_ops, undoable_mask

    # 对上面 返回”字典doable_ops“和返回”列表doable_ps“的引用
    def get_doable_ops(self, machine_id=None, return_list=False, shuffle_machine=True):
        if return_list:
            ret = self.get_doable_ops_in_list(machine_id, shuffle_machine)
        else:
            ret = self.get_doable_ops_in_dict(machine_id, shuffle_machine)
        return ret

    # 返回 状态、奖励、done
    def observe(self, reward='idle_time', return_doable=True):
        # A simple wrapper for JobManager's observe function
        # and return current time step reward r
        # check all jobs are done or not, then return done = True or False
        jobs_done = [job.job_done for _, job in self.job_manager.jobs.items()]
        # check jobs_done contains only True or False
        r = 0
        #self.proc_time_cp = np.copy(self.processing_time_matrix).astype(np.single)
        #self.LBs = calEndTimeLB(self,self.temp1, self.proc_time_cp)
        if np.prod(jobs_done) == 1:
            done = True
        else:
            done = False
        if reward == 'makespan':
            if done:
                r = -self.global_time
            else:
                r = 0
        # return reward as total sum of queues for all machines 等于所有机器的doable_ops的和
        # 也即相当于doable_ops 就是机器的队列数据
        elif reward == 'utilization':
            t_cost = self.machine_manager.cal_total_cost()
            r = -t_cost

        elif reward == 'idle_time':
            r = -float(len(self.machine_manager.get_idle_machines())) / float(self.num_machine)
        elif reward == 'LBs':
            r = - (self.LBs.max() - self.max_endTime)
            self.max_endTime = self.LBs.max()

        flag = False
        if flag:
            g = self.job_manager.observe(detach_done=self.detach_done)
        else:
            g = 0
        if flag:
            if return_doable:
                if self.use_surrogate_index:
                    do_ops_list, _ = self.get_doable_ops(return_list=True)
                    for n in g.nodes:
                        if n in do_ops_list:
                            job_id, op_id = self.job_manager.sur_index_dict[n]
                            m_id = self.job_manager[job_id][op_id].machine_id
                            g.nodes[n]['doable'] = True
                            g.nodes[n]['machine'] = m_id
                        else:
                            g.nodes[n]['doable'] = False
                            g.nodes[n]['machine'] = 0

        if return_doable:
            if self.use_surrogate_index:
                do_ops_list, undoable_mask = self.get_doable_ops(return_list=True)
                doable_mask = [bool(1-v) for v in do_ops_list]
                do_ops_list = do_ops_list[doable_mask]
                for job_id, job in self.job_manager.jobs.items():
                    for op in job.ops:
                        not_start_cond = not (op == job.ops[0])
                        not_end_cond = not (op == job.ops[-1])
                        done_cond = op.x['type'] == DONE_NODE_SIG
                        if op.sur_id in do_ops_list:
                            #job_id, op_id = self.job_manager.sur_index_dict[op.sur_id]
                            #m_id = self.job_manager[job_id][op_id].machine_id
                            op.doable_type = True
                            #op.machine_id = m_id
                        else:
                            op.doable_type = False
                            #op.machine_id = 0

        prt_fea = np.zeros(self.num_ops, dtype=np.single)
        com_fea = np.zeros(self.num_ops, dtype=np.single)
        rem_op_fea = np.zeros(self.num_ops, dtype=np.single)
        wait_time_fea = np.zeros(self.num_ops, dtype=np.single)
        rem_time_fea = np.zeros(self.num_ops, dtype=np.single)
        node_status_fea = np.zeros((self.num_ops, 3), dtype=np.single)
        node_status_single_fea = np.zeros(self.num_ops, dtype=np.single)

        if flag:
            for n in g.nodes:
                if g.nodes[n]["type"] == NOT_START_NODE_SIG:
                    node_status_fea[n] = [0]
                elif g.nodes[n]["type"] == PROCESSING_NODE_SIG:
                    node_status_fea[n] = [0]
                elif g.nodes[n]["type"] == DONE_NODE_SIG:
                    node_status_fea[n] = [1]
                else:
                    raise RuntimeError("Not supporting node type")
                prt_fea[n] = g.nodes[n]['processing_time']
                com_fea[n] = g.nodes[n]["complete_ratio"]
                rem_op_fea[n] = g.nodes[n]['remaining_ops']
                wait_time_fea[n] = g.nodes[n]['waiting_time']
                rem_time_fea[n] = g.nodes[n]["remain_time"]
        for job_id, job in self.job_manager.jobs.items():
            for op in job.ops:
                n = op.sur_id
                if op.node_status == NOT_START_NODE_SIG:
                    node_status_fea[n] = [1,0,0]
                    node_status_single_fea[n] = 0
                elif op.node_status == PROCESSING_NODE_SIG:
                    node_status_fea[n] = [0,1,0]
                    node_status_single_fea[n] = 0
                elif op.node_status == DONE_NODE_SIG:
                    node_status_fea[n] = [0,0,1]
                    node_status_single_fea[n] = 1
                else:
                    raise RuntimeError("Not supporting node type")
                prt_fea[n] = op.processing_time
                com_fea[n] = op.complete_ratio
                rem_op_fea[n] = op.remaining_ops
                wait_time_fea[n] = op.waiting_time
                rem_time_fea[n] = op.remaining_time
        node_status_single_fea = node_status_single_fea.reshape(self.num_jobs,self.num_steps)
        # for job_id, job in self.job_manager.jobs.items():
        #     for op in job.ops:
        #         not_start_cond = (op.node_status == NOT_START_NODE_SIG)
        #         processing_cond = (op.node_status == PROCESSING_NODE_SIG)
        #         done_cond = (op.node_status == DONE_NODE_SIG)
        #         # if not_start_cond:
        prt_fea_max = np.max(prt_fea)
        prt_fea_min = np.min(prt_fea)
        prt_fea = (prt_fea - prt_fea_min) / (prt_fea_max-prt_fea_min)
        #prt_fea = (prt_fea - np.mean(prt_fea))/np.std(prt_fea)

        # if np.max(wait_time_fea) == 0:
        #     wait_time_fea = wait_time_fea.reshape(self.num_ops, 1)
        # else:
        #     wait_time_fea = wait_time_fea.reshape(self.num_ops, 1) / np.max(wait_time_fea)
        #
        # if np.max(rem_time_fea) == 0:
        #     rem_time_fea = rem_time_fea.reshape(self.num_ops, 1)
        # else:
        #     #print("np.max(rem_time_fea):",np.max(rem_time_fea))
        #     rem_time_fea = rem_time_fea.reshape(self.num_ops, 1) / np.max(rem_time_fea)


        fea = np.concatenate((rem_op_fea.reshape(self.num_ops, 1)/self.num_machine,
                              # rem_time_fea,
                              com_fea.reshape(self.num_ops, 1),
                              #self.FDDMWKR.reshape(self.num_ops,1),
                              #prt_fea.reshape(self.num_ops, 1),
                              # com_fea.reshape(self.num_ops, 1)
                              node_status_fea.reshape(self.num_ops, 3)),axis=1)
        # rem_op_fea.reshape(self.num_ops, 1),
        # wait_time_fea.reshape(self.num_ops, 1)/np.max(wait_time_fea)), axis=1)
        # rem_time_fea.reshape(self.num_ops, 1)), axis=1)
        candidate, undoable_mask = self.get_doable_ops(return_list=True)
        # update adj matrix
        for m_id,m in self.machine_manager.machines.items():
            if len(m.done_ops) ==0 or len(m.done_ops)==1:
                pass
            else:
                op = m.done_ops[-1]
                pre_op = m.done_ops[-2]
                self.Adj[op.sur_id, pre_op.sur_id] = 1
                # self.Adj[pre_op.sur_id, op.sur_id] = 1
        # candidate = np.array(candidate,dtype=np.int64)
        return fea, self.Adj, g, float(r), candidate, undoable_mask, done

    def plot_graph(self, draw=True,
                   node_type_color_dict=None,
                   edge_type_color_dict=None,
                   half_width=None,
                   half_height=None,
                   **kwargs):

        g = self.job_manager.observe(self.detach_done)
        node_colors = get_node_color_map(g, node_type_color_dict)
        edge_colors = get_edge_color_map(g, edge_type_color_dict)

        if half_width is None:
            half_width = 30
        if half_height is None:
            half_height = 10

        num_horizontals = self.num_steps + 1
        num_verticals = self.num_jobs + 1

        def xidx2coord(x):
            return np.linspace(-half_width, half_width, num_horizontals)[x]

        def yidx2coord(y):
            return np.linspace(half_height, -half_height, num_verticals)[y]

        pos_dict = OrderedDict()
        for n in g.nodes:
            if self.use_surrogate_index:
                y, x = self.job_manager.sur_index_dict[n]
                pos_dict[n] = np.array((xidx2coord(x), yidx2coord(y)))
            else:
                pos_dict[n] = np.array((xidx2coord(n[1]), yidx2coord(n[0])))

        if kwargs is None:
            kwargs['figsize'] = (10, 5)
            kwargs['dpi'] = 300

        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(1, 1, 1)

        nx.draw(g, pos_dict,
                node_color=node_colors,
                edge_color=edge_colors,
                with_labels=True,
                ax=ax)
        if draw:
            plt.show()
        else:
            return fig, ax

    def draw_gantt_chart(self, path, benchmark_name, max_x):
        # Draw a gantt chart
        self.job_manager.draw_gantt_chart(path, benchmark_name, max_x)

    @staticmethod
    def _sample_jssp_graph(m, n):
        if not m % N_SEP == 0:
            m = int(N_SEP * (m // N_SEP))
            if m < N_SEP:
                m = N_SEP
        if not n % N_SEP == 0:
            n = int(N_SEP * (n // N_SEP))
            if n < N_SEP:
                n = N_SEP
        if m > n:
            raise RuntimeError(" m should be smaller or equal to n ")

        return jssp_sampling(m, n, 5, 100)
        # return jssp_sampling(m, n, 1, 5)

    @classmethod
    def from_path(cls, jssp_path):
        with open(jssp_path) as f:
            ms = []  # machines
            prts = []  # processing times
            for l in f:
                l_split = " ".join(l.split()).split(' ')
                if len(l_split)<=2:                   #pass the benchmark first line
                    continue
                m = l_split[0::2]
                prt = l_split[1::2]
                ms.append(np.array(m, dtype=int))
                prts.append(np.array(prt, dtype=float))

        ms = np.stack(ms)
        ms = ms + 1
        prts = np.stack(prts)
        # num_job, num_machine = ms.shape
        # name = jssp_path.split('/')[-1].replace('.txt', '')
        return ms,prts

#     @classmethod
#     def from_TA_path(cls, pt_path, m_path, **kwargs):
#         with open(pt_path) as f1:
#             prts = []
#             for l in f1:
#                 l_split = l.split(SEP)
#                 prt = [e for e in l_split if e != '']
#                 if NEW in prt[-1]:
#                     prt[-1] = prt[-1].split(NEW)[0]
#                 prts.append(np.array(prt, dtype=float))
#
#         with open(m_path) as f2:
#             ms = []
#             for l in f2:
#                 l_split = l.split(SEP)
#                 m = [e for e in l_split if e != '']
#                 if NEW in m[-1]:
#                     m[-1] = m[-1].split(NEW)[0]
#                 ms.append(np.array(m, dtype=int))
#
#         ms = np.stack(ms) - 1
#         prts = np.stack(prts)
#         num_job, num_machine = ms.shape
#         name = pt_path.split('/')[-1].replace('_PT.txt', '')
#
#         return cls(num_machines=num_machine,
#                    num_jobs=num_job,
#                    name=name,
#                    machine_matrix=ms,
#                    processing_time_matrix=prts,
#                    **kwargs)
#
#
# class NodeProcessingTimeSimulator(JSSPSimulator):
#     def reset(self):
#         self.job_manager = NodeProcessingTimeJobManager(self.machine_matrix,
#                                                         self.processing_time_matrix,
#                                                         embedding_dim=self.embedding_dim,
#                                                         use_surrogate_index=self.use_surrogate_index)
#         self.machine_manager = NodeProcessingTimeMachineManager(self.machine_matrix,
#                                                                 self.job_manager,
#                                                                 # self.delay,
#                                                                 self.verbose)
#         self.global_time = 0  # -1 matters a lot
