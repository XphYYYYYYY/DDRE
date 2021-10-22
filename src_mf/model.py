import numpy as np
import random as rd
from tqdm import tqdm
import time
from utils import read_edge_list, sign_prediction, node_classification
from sklearn.model_selection import train_test_split
import networkx as nx
from scipy import sparse
import math
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize


class ANSModel(object):
    def __init__(self, args):
        self.args = args
        self.model1_logs = {'epoch': [], 'auc': [], 'f1': []}
        self.model2_logs = {'epoch': [], 'auc': [], 'f1': []}
        self.model_mix_logs = {'epoch': [], 'auc': [], 'f1': []}
        self.out_emb1 = None
        self.in_emb1 = None
        self.out_emb2 = None
        self.in_emb2 = None
        self.S1 = None
        self.S2 = None
        if self.args.sign_prediction:
            self.edges, self.node_num = read_edge_list('./input/' + self.args.dataset + '.txt')
            self.train_edges, self.test_edges, = train_test_split(self.edges,
                                                                  test_size=self.args.test_size,
                                                                  random_state=self.args.split_seed)
        if self.args.node_classification:
            self.train_edges, self.node_num = read_edge_list('./input/' + self.args.dataset + '.txt')
        self.G = nx.DiGraph()
        self.setup()

    def setup(self):
        self.G.add_nodes_from(range(self.node_num))
        for edge in self.train_edges:
            self.G.add_edge(edge[0], edge[1], weight=edge[2])

    def calculate(self):
        np.seterr(divide='ignore', invalid='ignore')
        data_type = np.float32

        A = (nx.to_scipy_sparse_matrix(self.G, nodelist=list(range(self.node_num)), dtype=data_type,
                                       format='csc')).astype(data_type)
        D_list = (np.abs(A) @ np.ones((self.node_num, 1)))[:, 0]
        D_invList = D_list.copy()
        for i in range(D_invList.shape[0]):
            if D_invList[i] != 0:
                D_invList[i] = 1 / D_list[i]
        D_inv = sparse.diags(D_invList, format='csc', dtype=data_type)
        P = D_inv @ A
        P_abs = D_inv @ np.abs(A)

        P_sum = P @ np.ones((self.node_num, 1))
        tmp1 = P @ np.ones((self.node_num, 1))
        for i in range(self.args.h - 1):
            tmp1 = P @ tmp1
            P_sum += tmp1
        term1 = self.args.k * (self.args.h * np.ones((self.node_num, 1)) + P_sum)
        term1 = 1 / term1[:, 0]
        term1 = sparse.diags(term1, format='csc')
        term2 = self.args.k * (self.args.h * np.ones((self.node_num, 1)) - P_sum)
        term2 = 1 / term2[:, 0]
        term2 = sparse.diags(term2, format='csc')

        term3 = sparse.csc_matrix((self.node_num, self.node_num), dtype=data_type)
        term4 = sparse.csc_matrix((self.node_num, self.node_num), dtype=data_type)
        splitNum = math.ceil(self.node_num / self.args.slice_size)
        pbar = tqdm(total=splitNum, desc='Calculating summary matrix', ncols=100)
        for i in range(splitNum):
            pbar.update(1)
            start_index = i * self.args.slice_size
            if i != (splitNum - 1):
                end_index = (i + 1) * self.args.slice_size
            else:
                end_index = self.node_num
            P_sum = P[:, start_index: end_index]
            P_absSum = P_abs[:, start_index: end_index]
            tmp1 = P[:, start_index: end_index]
            tmp2 = P_abs[:, start_index: end_index]
            for j in range(self.args.h - 1):
                tmp1 = P @ tmp1
                tmp2 = P_abs @ tmp2
                P_sum += tmp1
                P_absSum += tmp2
            term3[:, start_index: end_index] = P_absSum + P_sum
            term4[:, start_index: end_index] = P_absSum - P_sum
        pbar.close()
        term3.eliminate_zeros()
        term4.eliminate_zeros()
        term5 = term1 @ term3
        term6 = term2 @ term4
        self.S1 = term5 / (1 / self.node_num)
        self.S1.data = np.log(self.S1.data)
        self.S1[self.S1 < 0] = 0
        self.S1.eliminate_zeros()
        self.S2 = term6 / (1 / self.node_num)
        self.S2.data = np.log(self.S2.data)
        self.S2[self.S2 < 0] = 0
        self.S2.eliminate_zeros()

        print('-' * 20, 0, '-' * 20)
        self.calculate_emb()
        self.evaluate(epoch=-1)

        beta = 1 / self.node_num
        for epoch in range(self.args.epoch_num):
            print('-' * 20, epoch + 1, '-' * 20)
            self.S1 = sparse.csc_matrix((self.node_num, self.node_num), dtype=data_type)
            for i in range(splitNum):
                start_index = i * self.args.slice_size
                if i != (splitNum - 1):
                    end_index = (i + 1) * self.args.slice_size
                else:
                    end_index = self.node_num
                tmp = self.S2[start_index: end_index, :].todense() + beta
                tmp = normalize(tmp, axis=1, norm='l1')
                tmp = term5[start_index: end_index, :] / tmp
                tmp = sparse.csr_matrix(tmp)
                self.S1[start_index: end_index, :] = tmp
            self.S1.data = np.log(self.S1.data)
            self.S1[self.S1 < 0] = 0
            self.S1.eliminate_zeros()

            self.S2 = sparse.csc_matrix((self.node_num, self.node_num), dtype=data_type)
            for i in range(splitNum):
                start_index = i * self.args.slice_size
                if i != (splitNum - 1):
                    end_index = (i + 1) * self.args.slice_size
                else:
                    end_index = self.node_num
                tmp = self.S1[start_index: end_index, :].todense() + beta
                tmp = normalize(tmp, axis=1, norm='l1')
                tmp = term6[start_index: end_index, :] / tmp
                tmp = sparse.csr_matrix(tmp)
                self.S2[start_index: end_index, :] = tmp
            self.S2.data = np.log(self.S2.data)
            self.S2[self.S2 < 0] = 0
            self.S2.eliminate_zeros()

            self.calculate_emb()
            self.evaluate(epoch=epoch)

    def calculate_emb(self):
        u, s, vt = svds(self.S1, self.args.dim)
        sqrtS = np.diag(np.sqrt(s))
        self.out_emb1 = u @ sqrtS
        self.out_emb1 = self.out_emb1[:, ::-1]
        self.in_emb1 = (sqrtS @ vt).T
        self.in_emb1 = self.in_emb1[:, ::-1]

        u, s, vt = svds(self.S2, self.args.dim)
        sqrtS = np.diag(np.sqrt(s))
        self.out_emb2 = u @ sqrtS
        self.out_emb2 = self.out_emb2[:, ::-1]
        self.in_emb2 = (sqrtS @ vt).T
        self.in_emb2 = self.in_emb2[:, ::-1]

    def evaluate(self, epoch):
        if self.args.sign_prediction:
            auc, f1 = sign_prediction(self.out_emb1, self.in_emb1, self.train_edges, self.test_edges)
            print('Branch for positive proximity: AUC %.3f, F1 %.3f' % (auc, f1))
            self.model1_logs['epoch'].append(epoch)
            self.model1_logs['auc'].append(auc)
            self.model1_logs['f1'].append(f1)

            auc, f1 = sign_prediction(self.out_emb2, self.in_emb2, self.train_edges, self.test_edges)
            print('Branch for negative proximity: AUC %.3f, F1 %.3f' % (auc, f1))
            self.model2_logs['epoch'].append(epoch)
            self.model2_logs['auc'].append(auc)
            self.model2_logs['f1'].append(f1)

            out_emb = np.hstack((self.out_emb1, self.out_emb2))
            in_emb = np.hstack((self.in_emb1, self.in_emb2))
            auc, f1 = sign_prediction(out_emb, in_emb, self.train_edges, self.test_edges)
            print('Two branches: AUC %.3f, F1 %.3f' % (auc, f1))
            self.model_mix_logs['epoch'].append(epoch)
            self.model_mix_logs['auc'].append(auc)
            self.model_mix_logs['f1'].append(f1)

        if self.args.node_classification:
            auc, f1 = node_classification(self.out_emb1, self.in_emb1, self.args.comm_path, self.args.test_size)
            print('Branch for positive proximity: AUC %.3f, F1 %.3f' % (auc, f1))
            self.model1_logs['epoch'].append(epoch)
            self.model1_logs['auc'].append(auc)
            self.model1_logs['f1'].append(f1)

            auc, f1 = node_classification(self.out_emb2, self.in_emb2, self.args.comm_path, self.args.test_size)
            print('Branch for negative proximity: AUC %.3f, F1 %.3f' % (auc, f1))
            self.model2_logs['epoch'].append(epoch)
            self.model2_logs['auc'].append(auc)
            self.model2_logs['f1'].append(f1)

            out_emb = np.hstack((self.out_emb1, self.out_emb2))
            in_emb = np.hstack((self.in_emb1, self.in_emb2))
            auc, f1 = node_classification(out_emb, in_emb, self.args.comm_path, self.args.test_size)
            print('Two branches: AUC %.3f, F1 %.3f' % (auc, f1))
            self.model_mix_logs['epoch'].append(epoch)
            self.model_mix_logs['auc'].append(auc)
            self.model_mix_logs['f1'].append(f1)

    def save_emb(self):
        dataset = self.args.dataset
        np.save(dataset + '_out_emb1', self.out_emb1)
        np.save(dataset + '_in_emb1', self.in_emb1)
        np.save(dataset + '_out_emb2', self.out_emb2)
        np.save(dataset + '_in_emb2', self.in_emb2)
        emb = np.hstack((self.out_emb1, self.in_emb1, self.out_emb2, self.in_emb2))
        np.save(dataset + 'emb', emb)
