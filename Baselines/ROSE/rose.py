import networkx as nx
from sklearn.model_selection import train_test_split
import numpy as np


dataset = 'Alpha'
test_ratio = 0.2

graph = nx.DiGraph()
edges = np.loadtxt('input/' + dataset + '.txt')
for i in range(edges.shape[0]):
    graph.add_edge(int(edges[i][0]), int(edges[i][1]), weight=edges[i][2])
edges = [[e[0], e[1], e[2]['weight']] for e in graph.edges.data()]
node_num = max(graph.nodes) + 1
graph.add_nodes_from(range(node_num))

seed = 123
train_edges, test_edges, = train_test_split(edges,
                                            test_size=test_ratio,
                                            random_state=seed)
with open('output/' + dataset + '_train_edges.txt', 'w') as f:
    for edge in train_edges:
        edge[2] = int(edge[2])
        edge = list(map(str, edge))
        f.write('\t'.join(edge) + '\n')
with open('output/' + dataset + '_test_edges.txt', 'w') as f:
    for edge in test_edges:
        edge[2] = int(edge[2])
        edge = list(map(str, edge))
        f.write('\t'.join(edge) + '\n')

with open('output/' + dataset + '_aug_train_edges.txt', 'w') as f2w:
    with open('output/' + dataset + '_train_edges.txt') as f:
        for line in f:
            line = line.strip().split('\t')
            line[2] = float(line[2])
            line = list(map(int, line))
            src = line[0]
            des = line[1]
            if line[2] > 0:
                f2w.write('\t'.join([str(src), str(node_num * 2 + des)]) + '\n')
                f2w.write('\t'.join([str(node_num + src), str(node_num * 3 + des)]) + '\n')
            elif line[2] < 0:
                f2w.write('\t'.join([str(src), str(node_num * 3 + des)]) + '\n')
                f2w.write('\t'.join([str(node_num + src), str(node_num * 2 + des)]) + '\n')
