import os
import networkx as nx

def dump_graph_txt(name, G):

    N = len(G.nodes)
    pos = nx.get_node_attributes(G, "pos")

    wd = os.path.dirname(os.path.abspath(__file__))
    txtfile = os.path.join(wd, name+".txt")

    with open(txtfile, "w") as f:
        print(N, file=f)
        print("\n".join([str(pos[i][0])+" "+str(pos[i][1]) for i in range(N)]), file=f)
        print(len(G.edges), file=f)
        print("\n".join([str(e[0])+" "+str(e[1]) for e in G.edges]), file=f)

def gen_graph(name, N=10, d=0.3):

    G = nx.random_geometric_graph(N, d)
    dump_graph_txt(name, G)

if __name__ == "__main__":
    for N in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        for d in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            for i in range(5):
                gen_graph("graphs/complex_graphs/complex-" + str(N) + "-" + str(d) + "-" + str(i), N=N, d=d)
                print(N, d, i)
