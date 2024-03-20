import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from collections import OrderedDict

from plotly.offline import plot

from pyjssp.configs import NOT_START_NODE_SIG, PROCESSING_NODE_SIG, DONE_NODE_SIG



class JobManager:
    def __init__(self,
                 machine_matrix,
                 processing_time_matrix,
                 use_surrogate_index=True):

        machine_matrix = machine_matrix.astype(int)
        processing_time_matrix = processing_time_matrix.astype(float)

        self.jobs = OrderedDict()

        # Constructing conjunctive edges
        for job_i, (m, pr_t) in enumerate(zip(machine_matrix, processing_time_matrix)):
            # m = m + 1  # To make machine index starts from 1
            self.jobs[job_i] = Job(job_i, m, pr_t)

        # Constructing disjunctive edges
        machine_index = list(set(machine_matrix.flatten().tolist()))
        for m_id in machine_index:
            job_ids, step_ids = np.where(machine_matrix == m_id)
            for job_id1, step_id1 in zip(job_ids, step_ids):
                op1 = self.jobs[job_id1][step_id1]
                ops = []
                for job_id2, step_id2 in zip(job_ids, step_ids):
                    if (job_id1 == job_id2) and (step_id1 == step_id2):
                        continue  # skip itself
                    else:
                        ops.append(self.jobs[job_id2][step_id2])
                op1.disjunctive_ops = ops

        self.use_surrogate_index = use_surrogate_index

        if self.use_surrogate_index:
            # Constructing surrogate indices:
            num_ops = 0
            self.sur_index_dict = dict()
            for job_id, job in self.jobs.items():
                for op in job.ops:
                    op.sur_id = num_ops
                    self.sur_index_dict[num_ops] = op._id
                    num_ops += 1

    def __call__(self, index):
        return self.jobs[index]

    def __getitem__(self, index):
        return self.jobs[index]



class Job:
    def __init__(self, job_id, machine_order, processing_time_order):
        self.job_id = job_id
        self.ops = list()
        self.processing_time = np.sum(processing_time_order)
        self.num_sequence = processing_time_order.size
        # Connecting backward paths (add prev_op to operations)
        cum_pr_t = 0
        for step_id, (m_id, pr_t) in enumerate(zip(machine_order, processing_time_order)):
            cum_pr_t += pr_t
            op = Operation(job_id=job_id, step_id=step_id, machine_id=m_id,
                           prev_op=None,
                           processing_time=pr_t,
                           complete_ratio=cum_pr_t / self.processing_time,
                           job=self)
            self.ops.append(op)
        for i, op in enumerate(self.ops[1:]):
            op.prev_op = self.ops[i]

        # Connecting forward paths (add next_op to operations)
        for i, node in enumerate(self.ops[:-1]):
            node.next_op = self.ops[i + 1]

    def __getitem__(self, index):
        return self.ops[index]

    # To check job is done or not using last operation's node status
    @property
    def job_done(self):
        if self.ops[-1].node_status == DONE_NODE_SIG:
            return True
        else:
            return False

    # To check the number of remaining operations
    @property
    def remaining_ops(self):
        c = 0
        for op in self.ops:
            if op.node_status != DONE_NODE_SIG:
                c += 1
        return c



class Operation:
    def __init__(self,
                 job_id,
                 step_id,
                 machine_id,
                 complete_ratio,
                 prev_op,
                 processing_time,
                 job,
                 next_op=None,
                 disjunctive_ops=None):

        self.job_id = job_id
        self.step_id = step_id
        self.job = job
        self._id = (job_id, step_id)
        self.machine_id = machine_id
        self.node_status = NOT_START_NODE_SIG
        self.complete_ratio = complete_ratio
        self.prev_op = prev_op
        # self.delayed_time = 0
        self.processing_time = int(processing_time)
        self.remaining_time = -1
        self.remaining_ops = self.job.num_sequence - (self.step_id + 1)
        self.waiting_time = 0
        self._next_op = next_op
        self._disjunctive_ops = disjunctive_ops

        self.next_op_built = False
        self.disjunctive_built = False
        self.built = False
        self.doable_type = False
        self.arrive_time = 0

    def __str__(self):  # 返回一个对象的描述信息
        return "job {} step {}".format(self.job_id, self.step_id)

    def processible(self):
        prev_none = self.prev_op is None
        if self.prev_op is not None:
            prev_done = self.prev_op.node_status is DONE_NODE_SIG
        else:
            prev_done = False
        return prev_done or prev_none

    @property
    def id(self):
        if hasattr(self, 'sur_id'):
            _id = self.sur_id
        else:
            _id = self._id
        return _id

    @property
    def disjunctive_ops(self):
        return self._disjunctive_ops

    @disjunctive_ops.setter
    def disjunctive_ops(self, disj_ops):
        for ops in disj_ops:
            if not isinstance(ops, Operation):
                raise RuntimeError("Given {} is not Operation instance".format(ops))
        self._disjunctive_ops = disj_ops
        self.disjunctive_built = True
        if self.disjunctive_built and self.next_op_built:
            self.built = True

    @property
    def next_op(self):
        return self._next_op

    @next_op.setter
    def next_op(self, next_op):
        self._next_op = next_op
        self.next_op_built = True
        if self.disjunctive_built and self.next_op_built:
            self.built = True

    @property
    def x(self):  # return node attribute
        not_start_cond = (self.node_status == NOT_START_NODE_SIG)
        # delayed_cond = (self.node_status == DELAYED_NODE_SIG)
        processing_cond = (self.node_status == PROCESSING_NODE_SIG)
        done_cond = (self.node_status == DONE_NODE_SIG)

        if not_start_cond:
            _x = OrderedDict()
            _x['id'] = self._id
            _x["type"] = self.node_status
            _x["complete_ratio"] = self.complete_ratio
            _x['processing_time'] = self.processing_time
            _x['remaining_ops'] = self.remaining_ops
            _x['waiting_time'] = self.waiting_time
            _x["remain_time"] = 0
        elif processing_cond or done_cond:
            _x = OrderedDict()
            _x['id'] = self._id
            _x["type"] = self.node_status
            _x["complete_ratio"] = self.complete_ratio
            _x['processing_time'] = self.processing_time
            _x['remaining_ops'] = self.remaining_ops
            _x['waiting_time'] = 0
            _x["remain_time"] = self.remaining_time
        # elif delayed_cond:
        #     raise NotImplementedError("delayed operation")
        else:
            raise RuntimeError("Not supporting node type")
        return _x
