import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm
import random as rd


def args_parser():
    parser = argparse.ArgumentParser(description="Run SLF.")
    parser.add_argument("--dataset",
                        nargs="?",
                        default="Alpha")
    parser.add_argument("--dim",
                        type=int,
                        default=16,
                        help="Dimension of latent factor vector.")
    parser.add_argument("--window_size",
                        type=int,
                        default=5,
                        help="Context window size.")
    parser.add_argument("--num_walks",
                        type=int,
                        default=80,
                        help="Walks per node.")
    parser.add_argument("--walk_len",
                        type=int,
                        default=40,
                        help="Length per walk.")
    parser.add_argument("--workers",
                        type=int,
                        default=4,
                        help="Number of threads used for random walking.")
    parser.add_argument("--test-size",
                        type=float,
                        default=0.2,
                        help="Test ratio.")
    parser.add_argument("--split-seed",
                        type=int,
                        default=123,
                        help="Random seed for splitting dataset.")
    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.01,
                        help="Learning rate.")
    parser.add_argument("--k",
                        type=int,
                        default=5,
                        help="Number of negative samples.")
    parser.add_argument("--epoch",
                        type=int,
                        default=1,
                        help="Epoch number.")

    return parser.parse_args()


def read_edge_list(args):
    G = nx.DiGraph()
    edges = np.loadtxt('input/' + args.dataset + '.txt')
    for i in range(edges.shape[0]):
        G.add_edge(int(edges[i][0]), int(edges[i][1]), weight=edges[i][2])
    edges = [[e[0], e[1], e[2]['weight']] for e in G.edges.data()]
    return edges, max(G.nodes) + 1


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
            while len(walk) < walk_len:
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
