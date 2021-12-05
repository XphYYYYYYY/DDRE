import networkx as nx
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse.linalg import svds
import math
from scipy import sparse


dataset = 'Alpha'
test_ratio = 0.2
graph = nx.Graph()
edges = np.loadtxt('input/' + dataset + '.txt')
for i in range(edges.shape[0]):
    graph.add_edge(int(edges[i][0]), int(edges[i][1]))
edges = [[e[0], e[1]] for e in graph.edges.data()]
node_num = max(graph.nodes) + 1
graph.add_nodes_from(range(node_num))

seed = 123
train_edges, _ = train_test_split(edges,
                                test_size=test_ratio,
                                random_state=seed)
graph = nx.Graph()
graph.add_nodes_from(range(node_num))
for edge in train_edges:
    graph.add_edge(edge[0], edge[1])

A = (nx.to_scipy_sparse_matrix(graph, dtype=np.float32, format='csc')).astype(np.float32)
D_list = (np.abs(A) @ np.ones((node_num, 1)))[:, 0]
vol = sum(D_list)
D_invList = D_list.copy()
for i in range(D_invList.shape[0]):
    if D_invList[i] != 0:
        D_invList[i] = 1 / D_list[i]
D_inv = sparse.diags(D_invList, format='csc', dtype=np.float32)
P_abs = D_inv @ np.abs(A)
M = np.zeros((node_num, node_num), dtype=np.float16)
sliceSize = 5000
splitNum = math.ceil(node_num / sliceSize)

b = 5
T = 5
for i in range(splitNum):
    print(i)
    startIndex = i * sliceSize
    if i != (splitNum - 1):
        endIndex = (i + 1) * sliceSize
    else:
        endIndex = node_num
    P_absSum = P_abs[:, startIndex: endIndex]
    tmp2 = P_abs[:, startIndex: endIndex]
    for j in range(T):
        if j == 0:
            continue
        tmp2 = P_abs @ tmp2
        P_absSum = P_absSum + tmp2
    tmp3 = np.zeros((endIndex - startIndex, endIndex - startIndex))
    for j in range(endIndex - startIndex):
        tmp3[j, j] = D_invList[startIndex + j]
    partM = vol / b / T * P_absSum @ tmp3
    partM[partM < 1] = 1
    partM = np.log(partM)
    M[:, startIndex: endIndex] = partM
M = sparse.csc_matrix(M, dtype=np.float32)

dim = 32
u, s, vt = svds(M, dim)
sqrtS = np.diag(np.sqrt(s))
emb = u @ sqrtS
emb = emb[:, ::-1]
np.save('output/emb', emb)
