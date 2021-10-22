import argparse
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
# from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from texttable import Texttable
from sklearn.model_selection import train_test_split
import warnings


def parameter_parser():
    """
    Parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run SLF.")
    parser.add_argument("--dataset",
                        nargs="?",
                        default="Slashdot",
                        help="Dataset name.")
    parser.add_argument("--comm-path",
                        nargs="?",
                        default="./input/community1.txt")
    parser.add_argument("--epoch-num",
                        type=int,
                        default=2,
                        help="Number of training epochs.")
    parser.add_argument("--dim",
                        type=int,
                        default=8,
                        help="Dimension of the representation.")
    parser.add_argument("--k",
                        type=int,
                        default=5,
                        help="Number of noise samples per data sample.")
    parser.add_argument("--h",
                        type=int,
                        default=4,
                        help="Highest order.")
    parser.add_argument("--split-seed",
                        type=int,
                        default=123,
                        help="Random seed for splitting dataset.")
    parser.add_argument("--test-size",
                        type=float,
                        default=0.2,
                        help="Test ratio. Default is 0.2.")
    parser.add_argument("--sign-prediction",
                        type=bool,
                        default=True,
                        help="Perform sign prediction or not.")
    parser.add_argument("--node-classification",
                        type=bool,
                        default=False,
                        help="perform link prediction or not.")
    parser.add_argument("--slice-size",
                        type=int,
                        default=5000,
                        help="Slice size. Default is 1000.")

    return parser.parse_args()


def read_edge_list(file_path):
    """
    Load edges from a txt file.
    """
    G = nx.DiGraph()
    edges = np.loadtxt(file_path)
    for i in range(edges.shape[0]):
        G.add_edge(int(edges[i][0]), int(edges[i][1]), weight=edges[i][2])
    edges = [[e[0], e[1], e[2]['weight']] for e in G.edges.data()]
    return edges, max(G.nodes) + 1  # index can start from 0.


def args_printer(args):
    """
    Print the parameters in tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    t = Texttable()
    l = [[k, args[k]] for k in args.keys()]
    l.insert(0, ["Parameter", "Value"])
    t.add_rows(l)
    print(t.draw())


def sign_prediction(out_emb, in_emb, train_edges, test_edges):
    """
    Evaluate the performance on the sign prediction task.
    :param out_emb: Outward embeddings.
    :param in_emb: Inward embeddings.
    :param train_edges: Edges for training the model.
    :param test_edges: Edges for test.
    """

    out_dim = out_emb.shape[1]
    in_dim = in_emb.shape[1]
    train_x = np.zeros((len(train_edges), (out_dim + in_dim) * 2))
    train_y = np.zeros((len(train_edges), 1))
    for i, edge in enumerate(train_edges):
        u = edge[0]
        v = edge[1]
        if edge[2] > 0:
            train_y[i] = 1
        else:
            train_y[i] = 0
        train_x[i, : out_dim] = out_emb[u]
        train_x[i, out_dim: out_dim + in_dim] = in_emb[u]
        train_x[i, out_dim + in_dim: out_dim * 2 + in_dim] = out_emb[v]
        train_x[i, out_dim * 2 + in_dim:] = in_emb[v]

    test_x = np.zeros((len(test_edges), (out_dim + in_dim) * 2))
    test_y = np.zeros((len(test_edges), 1))
    for i, edge in enumerate(test_edges):
        u = edge[0]
        v = edge[1]
        if edge[2] > 0:
            test_y[i] = 1
        else:
            test_y[i] = 0
        test_x[i, : out_dim] = out_emb[u]
        test_x[i, out_dim: out_dim + in_dim] = in_emb[u]
        test_x[i, out_dim + in_dim: out_dim * 2 + in_dim] = out_emb[v]
        test_x[i, out_dim * 2 + in_dim:] = in_emb[v]

    ss = StandardScaler()
    train_x = ss.fit_transform(train_x)
    test_x = ss.fit_transform(test_x)
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(train_x, train_y.ravel())
    test_y_score = lr.predict_proba(test_x)[:, 1]
    test_y_pred = lr.predict(test_x)
    auc_score = roc_auc_score(test_y, test_y_score, average='macro')
    macro_f1_score = f1_score(test_y, test_y_pred, average='macro')

    return auc_score, macro_f1_score


def sign_prediction_printer(logs):
    """
    Print the performance on sign prediction task in tabular format.
    :param logs: Logs about the evaluation.
    """
    t = Texttable()
    epoch_list = logs['epoch']
    auc_list = logs['auc']
    macrof1_list = logs['f1']
    l = [[epoch_list[i], auc_list[i], macrof1_list[i]] for i in range(len(epoch_list))]
    l.insert(0, ['Epoch', 'AUC', 'Macro-F1'])
    t.add_rows(l)
    # print(t.draw())
    print('*' * 5, 'Sign prediction', '*' * 5)
    print('-' * 5, 'AUC', '-' * 5)
    for auc in auc_list:
        print(round(auc, 3))
    print('-' * 5, 'Macro-F1', '-' * 5)
    for macrof1 in macrof1_list:
        print(round(macrof1, 3))
    print('*' * 10)


def node_classification(out_emb, in_emb, comm, test_size):
    emb = np.hstack((out_emb, in_emb))
    node_num = emb.shape[0]
    dim = emb.shape[1]
    auc_results = []
    f1_results = []
    for seed in range(5):
        train_set, test_set = train_test_split(range(node_num),
                                               test_size=test_size,
                                               random_state=seed)
        node2comm = {}
        with open(comm) as f:
            for line in f:
                line = line.strip().split('\t')
                line = list(map(int, line))
                node2comm[line[0]] = line[1]

        train_x = np.zeros((len(train_set), dim))
        train_y = np.zeros((len(train_set), 1))
        test_x = np.zeros((len(test_set), dim))
        test_y = np.zeros((len(test_set), 1))

        for i, node in enumerate(train_set):
            train_y[i] = node2comm[node]
            train_x[i] = emb[node]

        for i, node in enumerate(test_set):
            test_y[i] = node2comm[node]
            test_x[i] = emb[node]

        ss = StandardScaler()
        train_x = ss.fit_transform(train_x)
        test_x = ss.fit_transform(test_x)

        lr = LogisticRegression()

        def afunc(lr, trainX, trainY):
            lr.fit(trainX, trainY.ravel())
            return lr

        lr = afunc(lr, train_x, train_y)
        pred_prob = lr.predict_proba(test_x)
        pred_label = lr.predict(test_x)
        test_y = test_y[:, 0]
        AucScore = roc_auc_score(test_y, pred_prob, average='macro', multi_class='ovo')
        MacroF1Score = f1_score(test_y, pred_label, average='macro')
        auc_results.append(round(AucScore, 3))
        f1_results.append(round(MacroF1Score, 3))
    return sum(auc_results) / len(auc_results), sum(f1_results) / len(f1_results)


def node_classification_printer(logs):
    """
    Print the performance on sign prediction task in tabular format.
    :param logs: Logs about the evaluation.
    """
    t = Texttable()
    epoch_list = logs['epoch']
    auc_list = logs['auc']
    macrof1_list = logs['f1']
    l = [[epoch_list[i], auc_list[i], macrof1_list[i]] for i in range(len(epoch_list))]
    l.insert(0, ['Epoch', 'AUC', 'Macro-F1'])
    t.add_rows(l)
    # print(t.draw())
    print('*' * 5, 'Node classification', '*' * 5)
    print('-' * 5, 'AUC', '-' * 5)
    for auc in auc_list:
        print(round(auc, 3))
    print('-' * 5, 'Macro-F1', '-' * 5)
    for macrof1 in macrof1_list:
        print(round(macrof1, 3))
    print('*' * 10)
