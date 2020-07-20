import os
import sys
import argparse
import random
import networkx as nx
import matplotlib.pyplot as plt


def make_random_graph(seed=0):

    N = 20
    d = 0.3

    # generate random graph
    G = nx.random_geometric_graph(N, d, seed=seed)

    return G


def make_random_flows(G, seed=0):

    random.seed(seed)
    F = 10
    dG = nx.DiGraph(G)

    # pick random 1-hop flows
    # option 1: with non-overlapping senders
    nodes = random.sample(dG.nodes, F)
    flows = []
    for n in nodes:
        m = random.choice(list(dG.neighbors(n)))
        flows.append((n, m))
    # option 2: with possible overlapping senders
    # flows = random.sample(dG.edges, F)

    return flows


def dump_graph_pdf(name, G, flows, dump_pdf=False, show_graph=False):

    dG = nx.DiGraph(G)

    plt.figure()
    nx.draw(G, nx.get_node_attributes(G, "pos"), with_labels=True, edge_color="gray", style="dashed")
    nx.draw(dG, nx.get_node_attributes(G, "pos"), edgelist=flows, edge_color="red", width=2, arrows=True)

    if dump_pdf:
        wd = os.path.dirname(os.path.abspath(__file__))
        pdfname = os.path.join(wd, "graphs/"+name+".pdf")
        plt.savefig(pdfname)

    if show_graph:
        plt.show()


def dump_graph_txt(name, G, flows):
    
    N = len(G.nodes)
    pos = nx.get_node_attributes(G, "pos")

    wd = os.path.dirname(os.path.abspath(__file__))
    txtfile = os.path.join(wd, "graphs/"+name+".txt")

    with open(txtfile, "w") as f:
        print(N, file=f)
        print("\n".join([str(pos[i][0])+" "+str(pos[i][1])
                            for i in range(N)]), file=f)
        print(len(G.edges), file=f)
        print("\n".join([str(e[0])+" "+str(e[1])
                            for e in G.edges]), file=f)
        print(len(flows), file=f)
        print("\n".join([str(e[0])+" "+str(e[1]) for e in flows]), file=f)


def read_graph(name):

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
    nx.set_node_attributes(G, pos, "pos")

    return G, flows


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    args = parser.parse_args()

    # G = make_random_graph(seed=0)

    G, flows = read_graph("complex")
    dump_graph_pdf("complex", G, flows, dump_pdf=True)

    # for i in range(4):
    #     flows = make_random_flows(G, seed=i)
    #     dump_graph_pdf("complex-"+str(i), G, flows, dump_pdf=True)
    #     dump_graph_txt("complex-"+str(i), G, flows)
