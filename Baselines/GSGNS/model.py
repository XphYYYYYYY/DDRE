import networkx as nx
import random
import numpy as np
from collections import defaultdict
from utils import read_edge_list, parallel_generate_walks
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from tqdm import tqdm


class AdversarialNegativeSampling(object):
    def __init__(self, args):
        self.args = args
        self.edges, self.num_nodes, self.train_edges, self.test_edges, self.G, self.d_graph = self._setup()
        self.out_emb = np.random.rand(self.num_nodes, self.args.dim)
        self.in_emb = np.random.rand(self.num_nodes, self.args.dim)
        self.walks = self._generate_walks()

    def _setup(self):
        edges, num_nodes = read_edge_list(self.args)
        train_edges, test_edges = train_test_split(edges,
                                                   test_size=self.args.test_size,
                                                   random_state=self.args.split_seed)
        d_graph = defaultdict(dict)
        G = nx.DiGraph()
        for edge in train_edges:
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

        return edges, num_nodes, train_edges, test_edges, G, d_graph

    def _generate_walks(self):
        flatten = lambda l: [item for sublist in l for item in sublist]
        num_walks_lists = np.array_split(range(self.args.num_walks), self.args.workers)

        walk_results = Parallel(n_jobs=self.args.workers)(
            delayed(parallel_generate_walks)(self.d_graph,
                                             self.args.walk_len,
                                             len(num_walks),
                                             idx, ) for
            idx, num_walks in enumerate(num_walks_lists))

        return flatten(walk_results)

    def fit(self):

        def sigmoid(x):
            s = 1 / (1 + np.exp(-x))
            return s

        nodes = list(range(self.num_nodes))
        for epoch in range(self.args.epoch):
            random.shuffle(self.walks)
            pbar = tqdm(total=len(self.walks), desc='Optimizing', ncols=100)
            for walk in self.walks:
                pbar.update(1)
                walk_len = len(walk)
                for start in range(walk_len - 1):
                    u = walk[start]
                    sign = 1
                    context = walk[start + 1: min(start + self.args.window_size + 1, self.args.walk_len)]
                    pre_v = u
                    for v in context:
                        gradient_u = 0
                        sign *= self.G[pre_v][v]['polarity']
                        noisy_nodes = random.choices(nodes, k=self.args.k)
                        if sign > 0:
                            score = sigmoid(self.out_emb[u] @ self.in_emb[v])
                            gradient_u += (1 - score) * self.in_emb[v]
                            self.in_emb[v] += self.args.learning_rate * (1 - score) * self.out_emb[u]

                            for noise in noisy_nodes:
                                score = sigmoid(self.out_emb[u] @ self.in_emb[noise])
                                gradient_u += -score * self.in_emb[noise]
                                self.in_emb[noise] += self.args.learning_rate * -score * self.out_emb[u]
                            self.out_emb[u] += self.args.learning_rate * gradient_u
                        else:
                            score = sigmoid(self.out_emb[u] @ self.in_emb[v])
                            gradient_u += -score * self.in_emb[v]
                            self.in_emb[v] += self.args.learning_rate * - score * self.out_emb[u]

                            for noise in noisy_nodes:
                                score = sigmoid(self.out_emb[u] @ self.in_emb[noise])
                                gradient_u += (1 - score) * self.in_emb[noise]
                                self.in_emb[noise] += self.args.learning_rate * (1 - score) * self.out_emb[u]
                            self.out_emb[u] += self.args.learning_rate * gradient_u
                        pre_v = v
            pbar.close()

    def save_emb(self):
        dataset = self.args.dataset
        emb_path = 'output/'
        np.save(emb_path + dataset + '_out_emb', self.out_emb)
        np.save(emb_path + dataset + '_in_emb', self.in_emb)
