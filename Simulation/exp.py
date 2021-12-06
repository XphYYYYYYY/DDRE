from tqdm import tqdm
import numpy as np
import random as rd
from collections import defaultdict
import networkx as nx
from joblib import Parallel, delayed


def parallel_generate_walks(d_graph, walk_len, num_walks, cpu_num):
    walks = list()
    pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):
        pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        rd.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:
            walk = [source]
            while len(walk) < walk_len + 1:
                walk_options = d_graph[walk[-1]]['successors']
                if not walk_options:
                    break
                probabilities = d_graph[walk[-1]]['probabilities']
                walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
                walk.append(walk_to)
            if len(walk) > 2:
                walks.append(walk)
    pbar.close()
    return walks


def read_edge_list(dataset):
    G = nx.DiGraph()
    edges = np.loadtxt('input/' + dataset + '.txt')
    for i in range(edges.shape[0]):
        G.add_edge(int(edges[i][0]), int(edges[i][1]), weight=edges[i][2])
    edges = [[e[0], e[1], e[2]['weight']] for e in G.edges.data()]
    return edges, max(G.nodes) + 1


dataset = 'test'
edges, node_num = read_edge_list(dataset)
d_graph = defaultdict(dict)
G = nx.DiGraph()
G.add_nodes_from(range(node_num))
for edge in edges:
    if edge[2] > 0:
        G.add_edge(edge[0], edge[1], weight=edge[2], polarity=1)
    elif edge[2] < 0:
        G.add_edge(edge[0], edge[1], weight=abs(edge[2]), polarity=-1)
for node in G.nodes():
    unnormalized_weights = []
    succs = list(G.successors(node))

    if not succs:
        d_graph[node]['probabilities'] = []
        d_graph[node]['successors'] = []
    else:
        for succ in succs:
            weight = G[node][succ]['weight']
            unnormalized_weights.append(weight)
        unnormalized_weights = np.array(unnormalized_weights)
        d_graph[node]['probabilities'] = unnormalized_weights / unnormalized_weights.sum()
        d_graph[node]['successors'] = succs

num_walks = 10000
walk_len = 5
workers = 4
flatten = lambda l: [item for sublist in l for item in sublist]
num_walks_lists = np.array_split(range(num_walks), workers)

walk_results = Parallel(n_jobs=workers)(
    delayed(parallel_generate_walks)(d_graph,
                                     walk_len,
                                     len(num_walks),
                                     idx, ) for
    idx, num_walks in enumerate(num_walks_lists))
walk_results = flatten(walk_results)

target_node = 0
target_h = 2
pos_samples_nums = np.zeros(node_num)
for walk in walk_results:
    u = walk[0]
    if u == target_node:
        context = walk[1:]
        sign = 1
        pre_v = u
        h = 1
        for v in context:
            sign *= G[pre_v][v]['polarity']
            if h == target_h and sign > 0:
                pos_samples_nums[v] += 1
                break
            pre_v = v
            h += 1
print(pos_samples_nums)
print(pos_samples_nums / (num_walks * walk_len))

A = np.array(nx.adjacency_matrix(G, weight='polarity').todense())
print(A)
D_list = np.abs(A).sum(axis=1).astype(np.float32)
D_inv_list = D_list.copy()
for i in range(D_inv_list.shape[0]):
    if D_inv_list[i] != 0:
        D_inv_list[i] = 1 / D_list[i]
D_inv = np.diag(D_inv_list)
P = D_inv @ A
P_hat = D_inv @ np.abs(A)

P_pow = P
P_hat_pow = P_hat
for _ in range(target_h - 1):
    P_pow = P_pow @ P
    P_hat_pow = P_hat_pow @ P_hat

M = (P_hat_pow + P_pow) / 2

print(pos_samples_nums / num_walks)
print(M[target_node])
