import os
import sys
import random
import networkx as nx
import matplotlib.pyplot as plt

def dump_link_graph_pdf(name, G, dump_pdf=False, show_graph=False):

    plt.figure()
    nx.draw(G, nx.get_node_attributes(G, "pos"), with_labels=True)

    if dump_pdf:
        wd = os.path.dirname(os.path.abspath(__file__))
        pdfname = os.path.join(wd, "../graphs/link/"+name+".pdf")
        plt.savefig(pdfname)

    if show_graph:
        plt.show()

def dump_link_graph_txt(name, G):

    N = len(G.nodes)
    pos = nx.get_node_attributes(G, "pos")

    wd = os.path.dirname(os.path.abspath(__file__))
    txtfile = os.path.join(wd, "../graphs/link/"+name+".txt")

    with open(txtfile, "w") as f:
        print(N, file=f)
        print("\n".join([str(pos[i][0])+" "+str(pos[i][1]) for i in range(N)]), file=f)
        print(len(G.edges), file=f)
        print("\n".join([str(e[0])+" "+str(e[1]) for e in G.edges]), file=f)

def read_link_graph(name):

    wd = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(wd, "../graphs/link/"+name+".txt")
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
    except IOError as e:
        print(e.strerror, filename)
        sys.exit(1)
    N = int(lines[0].strip())
    pos = {i: [float(c) for c in lines[i+1].split()] for i in range(N)}
    M = int(lines[N+1].strip())
    edges = [[int(c) for c in lines[i+N+2].split()] for i in range(M)]

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)
    nx.set_node_attributes(G, pos, "pos")

    return G, N

def read_node_graph(name):

    wd = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(wd, "../graphs/node/"+name+".txt")
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
    except IOError as e:
        print(e.strerror, filename)
        sys.exit(1)
    
    N = int(lines[0].strip())
    pos = {i: [float(c) for c in lines[i+1].split()] for i in range(N)}
    M = int(lines[N+1].strip())
    edges = [[int(c) for c in lines[i+N+2].split()] for i in range(M)]
    F = int(lines[N+M+2].strip())
    flows = [[int(c) for c in lines[M+N+i+3].split()] for i in range(F)]

    return None, F

def gen_link_graph(name, threshold=0.3, sigma=0):
    
    # randomly generate graph
    d = random.normalvariate(threshold, sigma)
    d = max(d, 0)
    G = nx.random_geometric_graph(10, d)
    print("Generate G(10, {})".format(d))
    dump_link_graph_txt(name, G)

if __name__ == "__main__":

    G, N = read_link_graph("complex2")
    dump_link_graph_pdf("complex2", G, dump_pdf=True)

    # G = nx.random_geometric_graph(10, 0.3, seed=0)
    # dump_graph_pdf("complex2", G, dump_pdf=True)
    # dump_graph_txt("complex", G)
