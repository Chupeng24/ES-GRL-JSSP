import networkx as nx
import numpy as np

# def gen_adj_matrix(env, num_ops, num_steps):
#     # create Adj_matrix for all succeeding node embedding
#     Adj_suc_all = np.zeros([num_ops, num_ops], dtype=np.single)
#     temp_val = 1
#     for Adj_row, _ in enumerate(Adj_suc_all):
#         if Adj_row % num_steps == 0 and Adj_row != 0:
#             temp_val += 1
#         for Adj_col in range(Adj_row + 1, temp_val * num_steps):
#             Adj_suc_all[Adj_row][Adj_col] = 1.0
#
#     # create Adj_matrix for all precedent node embedding
#     Adj_pre_all = np.zeros([num_ops, num_ops], dtype=np.single)
#     temp_val = 0
#     for Adj_row, _ in enumerate(Adj_pre_all):
#         if Adj_row % num_steps == 0 and Adj_row != 0:
#             temp_val += 1
#         for Adj_col in range(temp_val * num_steps, Adj_row):
#             Adj_pre_all[Adj_row][Adj_col] = 1.0
#
#     # create Adj_matrix for succeeding node embedding
#     Adj_suc = np.zeros([num_ops, num_ops], dtype=np.single)
#     last_col = np.arange(start=0, stop=num_ops, step=1).reshape(env.num_jobs, -1)[:, -1]
#     for Adj_row, _ in enumerate(Adj_suc):
#         # Adj_suc[Adj_row][Adj_row] = 1.0
#         if Adj_row != num_ops - 1:
#             Adj_suc[Adj_row][Adj_row + 1] = 1.0
#     Adj_suc[last_col] = 0
#
#     # create Adj_matrix for all precedent node embedding
#     Adj_pre = np.zeros([num_ops, num_ops], dtype=np.single)
#     first_col = np.arange(start=0, stop=num_ops, step=1).reshape(env.num_jobs, -1)[:, 0]
#     for Adj_row, _ in enumerate(Adj_pre):
#         if Adj_row != 0:
#             Adj_pre[Adj_row][Adj_row - 1] = 1.0
#     Adj_pre[first_col] = 0
#
#     # create Adj_matrix for graph level embedding
#     Adj_all = np.ones([num_ops, num_ops], dtype=np.single)
#
#     # create Adj_matrix for disjunctive node embedding
#     Adj_disjunctive = np.zeros([num_ops, num_ops], dtype=np.single)
#     machine_index = env.machine_matrix.flatten().tolist()
#     for Adj_row, item in enumerate(Adj_disjunctive):
#         for Adj_col, _ in enumerate(item):
#             if machine_index[Adj_col] == machine_index[Adj_row]:
#                 item[Adj_col] = 1.0
#     for index in range(num_ops):
#         Adj_disjunctive[index][index] = 0
#
#     return Adj_pre, Adj_suc, Adj_all, Adj_disjunctive

def gen_adj_matrix(env, num_ops, num_steps):
    # initialize adj matrix
    conj_nei_up_stream = np.eye(num_ops,k=-1,dtype=np.single)
    conj_nei_low_stream = np.eye(num_ops,k=1,dtype=np.single)
    self_as_nei = np.eye(num_ops,dtype=np.single)
    # first column does not have upper stream conj_nei
    first_col = np.arange(start=0, stop=num_ops, step=1).reshape(num_steps, -1)[:, 0]
    last_col = np.arange(start=0, stop=num_ops, step=1).reshape(num_steps, -1)[:, -1]
    conj_nei_up_stream[first_col]=0
    conj_nei_low_stream[last_col]=0

    # create Adj_matrix for disjunctive node embedding
    # Adj_disjunctive = np.zeros([num_ops, num_ops], dtype=np.single)
    # machine_index = env.machine_matrix.flatten().tolist()
    # for Adj_row, item in enumerate(Adj_disjunctive):
    #     for Adj_col, _ in enumerate(item):
    #         if machine_index[Adj_col] == machine_index[Adj_row]:
    #             item[Adj_col] = 1.0
    # for index in range(num_ops):
    #     Adj_disjunctive[index][index] = 0

    adj = conj_nei_up_stream  + self_as_nei  # + Adj_disjunctive # + conj_nei_low_stream

    return adj

# if __name__ == '__main__':
#     s = FT06(verbose=True)
#     print(s.num_op)
#     Adj_pre, Adj_suc, Adj_all, Adj_disjunctive = gen_adj_matrix(s, s.num_op, s.num_steps)
#     print("end")
