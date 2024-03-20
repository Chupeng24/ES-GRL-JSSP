import numpy as np

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
    adj = conj_nei_up_stream  + self_as_nei  # Adj_disjunctive # + conj_nei_low_stream

    return adj
