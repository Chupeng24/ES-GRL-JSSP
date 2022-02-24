import numpy as np
import random
from pyjssp.benchmarks import FT06

np.random.seed(1)
random.seed(1)

# s = Simulator(4, 4, delay=False)
s = FT06(verbose=True)
while True:
    s.flush_trivial_ops()

    g, _, done = s.observe()
    if not done:
        s.transit()

    if done:
        print(s.global_time)
        s.plot_graph()
        s.job_manager.draw_gantt_chart("random_test_result2.html", "Ft06 Scheduling result", 100)
        break
