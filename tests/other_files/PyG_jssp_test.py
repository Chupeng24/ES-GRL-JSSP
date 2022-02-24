import numpy as np
import random
from pyjssp.benchmarks import FT06
import torch_geometric.utils
from pyjssp.utils import pprint_graph

np.random.seed(1)
random.seed(1)

# s = Simulator(4, 4, delay=False)
s = FT06(verbose=True)

while True:
    # s.flush_trivial_ops()
    s.transit()
    s.flush_trivial_ops()

    fea, adj_matrix, g, _, done = s.observe()
    for n in g.nodes():
        print('{}:'.format(n))
        print(g.nodes[n])
    data = torch_geometric.utils.from_networkx(G=g,
                                               group_node_attrs=['id'],
                                               group_edge_attrs=['type'])
    s.plot_graph()
    print("########################################################")

    if done:
        print(s.global_time)
        s.plot_graph()
        s.job_manager.draw_gantt_chart("random_test_result.html", "Ft06 Scheduling result", 100)
