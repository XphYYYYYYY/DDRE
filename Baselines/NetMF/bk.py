import numpy as np
import math
from scipy.sparse.linalg import svds
import networkx as nx
import time
from scipy import sparse


DATA_DIR = 'input_ANS/'
DATASET_NAME = 'Slashdot'
REMAIN = 80
H = 5  # Highest order
K = 5  # Number of negative samples
DIM = 32  # Dimension of emb

for seed in range(1, 11, 20):
    start = time.time()
    print(seed)
    seed = str(seed)
    G = nx.Graph()
    data_dir = DATA_DIR + DATASET_NAME + '/remain=' + str(REMAIN) + '/seed=' + seed
    with open(data_dir + '/train_edges.txt') as f:
        for line in f:
            line = line.strip().split('\t')
            line = list(map(int, line))
            G.add_edge(line[0], line[1])
    with open(data_dir + '/test_edges.txt') as f:
        for line in f:
            line = line.strip().split('\t')
            line = list(map(int, line))
            G.add_edge(line[0], line[1])
    numNodes = max(G.nodes) + 1
    numNodes *= 4

    G = nx.Graph()
    G.add_nodes_from(list(range(numNodes)))
    with open(data_dir + '/aug_train_edges.txt') as f:
        for line in f:
            line = line.strip().split('\t')
            line = list(map(int, line))
            G.add_edge(line[0], line[1])

    A = (nx.to_scipy_sparse_matrix(G, dtype=np.float32, format='csc')).astype(np.float32)
    D_list = (np.abs(A) @ np.ones((numNodes, 1)))[:, 0]
    vol = sum(D_list)
    D_invList = D_list.copy()
    for i in range(D_invList.shape[0]):
        if D_invList[i] != 0:
            D_invList[i] = 1 / D_list[i]
    D_inv = sparse.diags(D_invList, format='csc', dtype=np.float32)
    P_abs = D_inv @ np.abs(A)

    M = np.zeros((numNodes, numNodes), dtype=np.float16)
    sliceSize = 5000
    splitNum = math.ceil(numNodes / sliceSize)

    for i in range(splitNum):
        print(i)
        startIndex = i * sliceSize
        if i != (splitNum - 1):
            endIndex = (i + 1) * sliceSize
        else:
            endIndex = numNodes
        P_absSum = P_abs[:, startIndex: endIndex]
        tmp2 = P_abs[:, startIndex: endIndex]
        # print('stage -1')
        for j in range(H):
            if j == 0:
                continue
            tmp2 = P_abs @ tmp2
            P_absSum = P_absSum + tmp2
        tmp3 = np.zeros((endIndex - startIndex, endIndex - startIndex))
        for j in range(endIndex - startIndex):
            tmp3[j, j] = D_invList[startIndex + j]
        # print('stage 0')
        partM = vol / K / H * P_absSum @ tmp3
        # print('stage 1')
        partM[partM < 1] = 1
        partM = np.log(partM)
        # print('stage 2')
        M[:, startIndex: endIndex] = partM

    M = sparse.csc_matrix(M, dtype=np.float32)

    # M = M.astype(np.float32)

    u, s, vt = svds(M, 32)

    sqrtS = np.diag(np.sqrt(s))
    emb = u @ sqrtS
    emb = emb[:, ::-1]

    end = time.time()
    print("Execution Time: ", end - start)
    print('-' * 20)

    np.save('C:/Users/wenbin/PycharmProjects/AdversarialNegativeSampling/emb/' + DATASET_NAME + '/remain=' + str(REMAIN) + '/seed=' + str(seed) + '/ROSE/emb', emb)
