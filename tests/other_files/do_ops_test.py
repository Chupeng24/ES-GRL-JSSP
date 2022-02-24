from pyjssp.simulators import Simulator

# OK
if __name__ == "__main__":
    s = Simulator(5, 5)
    _, _, g, r, done = s.observe()

    for n in g.nodes:
        print(n, g.nodes[n])
    s.plot_graph()
