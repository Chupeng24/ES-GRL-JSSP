import random
from collections import OrderedDict
import numpy as np
from pyjssp.configs import (PROCESSING_NODE_SIG,
                            DONE_NODE_SIG)


class MachineManager:
    def __init__(self,
                 machine_matrix,
                 job_manager,
                 # delay=True,
                 proctime_std,
                 prac_proc_time_matrix,
                 temp1,
                 mbrk_Ag,
                 brk_rep_time_table,
                 verbose=False):

        machine_matrix = machine_matrix.astype(int)
        # Parse machine indices
        self.machine_index = list(set(machine_matrix.flatten().tolist()))

        # Global machines dict
        self.machines = OrderedDict()
        for m_id in self.machine_index:
            job_ids, step_ids = np.where(machine_matrix == m_id)
            possible_ops = []
            for job_id, step_id in zip(job_ids, step_ids):
                possible_ops.append(job_manager[job_id][step_id])

            self.machines[m_id] = Machine(m_id, possible_ops, verbose,
                                          proctime_std,
                                          prac_proc_time_matrix,
                                          temp1,
                                          mbrk_Ag,
                                          brk_rep_time_table)

    def do_processing(self, env,  t, shor_interval):
        for _, machine in self.machines.items():
            machine.do_processing(env, t, shor_interval)

    def load_op(self, machine_id, op, t):
        self.machines[machine_id].load_op(op, t)

    def __getitem__(self, index):
        return self.machines[index]

    def get_available_machines(self, shuffle_machine=True):
        m_list = []
        for _, m in self.machines.items():
            if m.available():
                m_list.append(m)

        if shuffle_machine:
            m_list = random.sample(m_list, len(m_list))

        return m_list

    # get idle machines' list
    def get_idle_machines(self):
        m_list = []
        for _, m in self.machines.items():
            if m.current_op is None and not m.work_done():
                m_list.append(m)
        return m_list

    # calculate the length of queues for all machines
    def cal_total_cost(self):
        c = 0
        for _, m in self.machines.items():
            c += len(m.doable_ops_no_delay)
        return c

    # update all cost functions of machines
    def update_cost_function(self, cost):
        for _, m in self.machines.items():
            m.cost += cost

    def get_machines(self):
        m_list = [m for _, m in self.machines.items()]
        return random.sample(m_list, len(m_list))

    # def all_delayed(self):
    #     return np.product([m.delayed_op is not None for _, m in self.machines.items()])

    def fab_stuck(self):
        # All machines are not available and All machines are delayed.
        all_machines_not_available_cond = not self.get_available_machines()
        # all_machines_delayed_cond = self.all_delayed()
        return all_machines_not_available_cond  # and all_machines_delayed_cond


class Machine:
    def __init__(self, machine_id, possible_ops, verbose, proctime_std,
                 prac_proc_time_matrix, temp1, mbrk_Ag, brk_rep_time_table):
        self.machine_id = machine_id
        self.possible_ops = possible_ops
        self.remain_ops = possible_ops
        self.current_op = None
        # self.delayed_op = None
        self.prev_op = None
        self.remaining_time = 0
        self.done_ops = []
        self.num_done_ops = 0
        self.cost = 0
        self.verbose = verbose
        self.proctime_std = proctime_std
        self.prac_proc_time_matrix = prac_proc_time_matrix
        self.temp1 = temp1
        self.mbrk_Ag = mbrk_Ag
        self.brk_rep_time_table = brk_rep_time_table
        self.normal_flag = True
        # set for machine breakdown situation
        if self.mbrk_Ag is not None and self.mbrk_Ag > 0:
            self.mbdatime = 0
            self.mbd_inf_op_id = []
            self.temp_op_adj = None
        self.total_proc = 0

    def __str__(self):
        return "Machine {}".format(self.machine_id)

    def available(self):
        future_work_exist_cond = bool(self.doable_ops())
        currently_not_processing_cond = self.current_op is None
        #  not_wait_for_delayed_cond = not self.wait_for_delayed()
        ret = future_work_exist_cond and currently_not_processing_cond and self.normal_flag  # and not_wait_for_delayed_cond
        return ret

    def doable_ops(self):
        # doable_ops are subset of remain_ops.
        # some ops are doable when the prev_op is 'done' or 'processing' or 'start'
        doable_ops = []
        for op in self.remain_ops:
            prev_start = op.prev_op is None
            if prev_start:
                doable_ops.append(op)
            else:
                prev_done = op.prev_op.node_status == DONE_NODE_SIG
                if prev_done:
                    doable_ops.append(op)
                else:
                    pass
        return doable_ops

    @property
    def doable_ops_id(self):
        doable_ops_id = []
        doable_ops = self.doable_ops()
        for op in doable_ops:
            doable_ops_id.append(op.id)

        return doable_ops_id

    @property
    def doable_ops_no_delay(self):
        doable_ops = []
        for op in self.remain_ops:
            prev_start = op.prev_op is None
            if prev_start:
                doable_ops.append(op)
            else:
                prev_done = op.prev_op.node_status == DONE_NODE_SIG
                if prev_done:
                    doable_ops.append(op)
        return doable_ops

    def work_done(self):
        return not self.remain_ops

    def load_op(self, t, op):
        # Procedures for double-checkings)
        # ignore input when the machine is not available
        if not self.available():
            raise RuntimeError("Machine {} is not available".format(self.machine_id))

        # ignore when input op's previous op is not done yet:
        if not op.processible():
            raise RuntimeError("Operation {} is not processible yet".format(print(op)))

        if op not in self.possible_ops:
            raise RuntimeError("Machine {} can't perform ops {}{}".format(self.machine_id,
                                                                          op.job_id,
                                                                          op.step_id))

        if self.verbose:
            print("[LOAD] / Machine {} / {} on at {}".format(self.machine_id, op, t))

        # Update operation's attributes
        op.node_status = PROCESSING_NODE_SIG
        op.remaining_time = op.processing_time

        if self.proctime_std:
            job_id, step_id = op._id
            op.remaining_time = self.prac_proc_time_matrix[job_id][step_id]
            # if op.remaining_time <= 0:
            #     op.remaining_time = 1
        else:
            op.remaining_time = op.processing_time
        op.start_time = t

        # Update machine's attributes
        self.current_op = op

        if self.proctime_std:
            self.remaining_time = op.remaining_time
        else:
            self.remaining_time = op.processing_time
        self.remain_ops.remove(self.current_op)
        if self.remaining_time <= 0:   #orb7 instance have op processing_time equals zero
            self.unload(t)

    def unload(self, t):
        if self.verbose:
            print("[UNLOAD] / Machine {} / Op {} / t = {}".format(self.machine_id, self.current_op, t))
        self.current_op.node_status = DONE_NODE_SIG
        self.current_op.end_time = t
        if self.proctime_std:
            job_id, step_id = self.current_op._id
            self.temp1[job_id,step_id] = t
        if self.current_op.next_op_built:
            self.current_op.next_op.arrive_time = t
        self.done_ops.append(self.current_op)
        self.num_done_ops += 1
        self.prev_op = self.current_op
        self.current_op = None
        self.remaining_time = -1

    def do_processing(self, env, t, shor_interval):
        if self.normal_flag:
            if self.remaining_time > 0:  # When machine do some operation
                if self.current_op is not None:
                    self.current_op.remaining_time -= shor_interval
                    self.total_proc += shor_interval
                    if self.current_op.remaining_time <= 0:
                        if self.current_op.remaining_time < 0:
                            raise RuntimeWarning("Negative remaining time observed")
                        if self.verbose:
                            print("[OP DONE] : / Machine  {} / Op {}/ t = {} ".format(self.machine_id, self.current_op, t))
                        self.unload(t)
                self.remaining_time -= shor_interval

        doable_ops = self.doable_ops()
        if doable_ops:
            if self.normal_flag:
                for op in doable_ops:
                    op.waiting_time += shor_interval
        else:
            pass
        if self.mbrk_Ag:
            if self.normal_flag == False:
                if self.current_op is not None or len(doable_ops)>0:
                    self.mbdatime += shor_interval
                    # if self.current_op is not None:
                    #     print("self.current_op is not None:", self.current_op._id)
                    #     print("self.current process time:", self.current_op.processing_time)
                    #     print("self.current remain time:",self.current_op.remaining_time)
                    # if len(doable_ops)>0:
                    #     print("doable is not None:")
                    #     for op in doable_ops:
                    #         print(op._id)
                    # print("machine_id:",self.machine_id,
                    #       "time:",t,
                    #       "interval:",shor_interval)
                    # print("______________________________________")
        if self.mbrk_Ag:
            flag = 0
            for idx, val in enumerate(self.brk_rep_time_table[self.machine_id-1]): # machin_id start form 1
                if t < val:
                    flag = idx
                    self.trans_interval = val - t
                    break
            if flag % 2 == 0:    # judge wether machine
                self.normal_flag = True
                if self.temp_op_adj is not None:
                    env.Adj[self.mbd_inf_op_id] = self.temp_op_adj
                    self.temp_op_adj = None
            else:
                self.normal_flag = False
                self.mbd_inf_op_id = []
                if self.current_op is not None:
                    for op in self.current_op.job.ops[self.current_op.step_id:]:
                        self.mbd_inf_op_id.append(op.sur_id)

                if self.remain_ops is not None:
                    for dis_op in self.remain_ops:
                        for op in dis_op.job.ops[dis_op.step_id:]:
                            self.mbd_inf_op_id.append(op.sur_id)

                self.temp_op_adj = env.Adj[self.mbd_inf_op_id]
                env.Adj[self.mbd_inf_op_id] = np.zeros((len(self.mbd_inf_op_id), env.Adj.shape[1]))

    def transit(self, t, a):
        if self.available():  # Machine is ready to process.
            if a.processible():  # selected action is ready to be loaded right now.
                self.load_op(t, a)
        else:
            raise RuntimeError("Access to not available machine")
