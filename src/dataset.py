import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

edges_data = "data/E-coli/Escherichia-coli_edge-list.txt"
labels_data = "data/E-coli/Escherichia-coli.txt"

# Nodes
with open(edges_data) as edgelist:
    nb_nodes = int(edgelist.readline().strip("Nb_nodes:"))
G = nx.Graph()
G.add_nodes_from(range(nb_nodes))

# Edges
with open(edges_data) as edgelist:
    partial_graph = nx.read_edgelist(edgelist)
for edge in partial_graph.edges():
    G.add_edge(int(edge[0]), int(edge[1]))

# Labels
with open(labels_data) as labels:
    labels = labels.readlines()[1:]
    labels = [label.strip() for label in labels]
    labels = np.array(labels)
assert len(labels) == len(G.nodes())
for i, label in enumerate(labels):
    G.nodes[i]["label"] = label

nx.draw(G)
plt.show()
