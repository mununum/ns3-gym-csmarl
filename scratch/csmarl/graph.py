import os
import sys
import argparse
import random
import networkx as nx
import matplotlib.pyplot as plt


def make_random_graph(dump_txt=True, dump_pdf=True, show_graph=True, seed=0):

    N = 20
    d = 0.3
    n_flows = 10
    random.seed(seed)

    # generate random graph
    G = nx.random_geometric_graph(N, d, seed=seed)
    dG = nx.DiGraph(G)
    pos = {i: G.nodes[i]["pos"] for i in range(N)}

    # pick random 1-hop flows (with non-overlapping senders)
    nodes = random.sample(dG.nodes, n_flows)
    flows = []
    for n in nodes:
        m = random.choice(list(dG.neighbors(n)))
        flows.append((n, m))
    # (with possible overlapping senders)
    # flows = random.sample(dG.edges, n_flows)

    # dump graph description
    if dump_txt:
        with open("graphs/complex.txt", "w") as f:
            print(N, file=f)
            print("\n".join([str(pos[i][0])+" "+str(pos[i][1])
                            for i in range(N)]), file=f)
            print(len(G.edges), file=f)
            print("\n".join([str(e[0])+" "+str(e[1]) for e in G.edges]), file=f)
            print(n_flows, file=f)
            print("\n".join([str(e[0])+" "+str(e[1]) for e in flows]), file=f)

    nx.draw(G, pos, with_labels=True, edge_color="gray", style="dashed")
    nx.draw(dG, pos, edgelist=flows,
            edge_color="red", width=2, arrows=True)

    if show_graph:
        plt.show()
    if dump_pdf:
        plt.savefig("random_topology.pdf")


def read_graph(name, show_graph=False, dump_pdf=False):
    wd = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(wd, "graphs/"+name+".txt")
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
    except IOError as e:
        print(e.strerror)
        sys.exit(1)
    N = int(lines[0].strip())
    pos = {i: [float(c) for c in lines[i+1].split()] for i in range(N)}
    M = int(lines[N+1].strip())
    edges = [[int(c) for c in lines[i+N+2].split()] for i in range(M)]
    F = int(lines[N+M+2].strip())
    flows = [[int(c) for c in lines[M+N+i+3].split()] for i in range(F)]

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)
    dG = nx.DiGraph(G)

    nx.draw(G, pos, with_labels=True, edge_color="gray", style="dashed")
    nx.draw(dG, pos, edgelist=flows,
            edge_color="red", width=2, arrows=True)

    if show_graph:
        plt.show()
    if dump_pdf:
        pdfname = os.path.join(wd, "graphs/"+name+".pdf")
        plt.savefig(pdfname)

    return F

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    args = parser.parse_args()

    # make_random_graph(show_graph=True, dump_txt=False, dump_pdf=False, seed=args.seed)

    read_graph("fim", show_graph=True, dump_pdf=False)


# G = nx.petersen_graph()
# plt.subplot(121)
# nx.draw(G, with_labels=True, font_weight='bold')
# plt.subplot(122)
# nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
# plt.show()
