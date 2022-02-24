import numpy as np
import random
from pyjssp.benchmarks import FT06, FT10
from pyjssp.utils import pprint_graph
#from pyjssp.simulators import Simulator

np.random.seed(1)
from pyjssp.utils import Timer

random.seed(1)

# s = Simulator(4, 4)
s = FT06(verbose=True)
timer = Timer(name="scheduling")

# for job_id, job in s.job_manager.jobs.items():
#     for op in job.ops:
#         print(op.sur_id)

with timer:
    while True:
        # s.flush_trivial_ops()
        # s.transit()
        # s.flush_trivial_ops()
        #
        # fea, adj_matrix, g, _, _,_, done = s.observe()
        fea, adj_matrix, g, _, _,_, done = s.step()
        # for n in g.nodes():
        #     print('{}:'.format(n))
        #     print(g.nodes[n])
        # s.plot_graph()
        # print("###########################################")
        if done:
            print(s.global_time)
            # s.plot_graph()
            # s.job_manager.draw_gantt_chart("random_test_result.html","Ft06 Scheduling result",100)
            for m_id,m in s.machine_manager.machines.items():
                print(m_id,"#########################")
                for done_op in m.done_ops:
                    print(done_op.sur_id)
            break
