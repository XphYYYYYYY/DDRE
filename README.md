# Dual-branch Density Ratio Estimation (DDRE)
Implementation of dual-branch density ratio estimation in Python.
The paper is submiited to WWW'22 and under review.

## Overview
This repository is organised as follows:
- `input/` contains the used datasets.
  - `Alpha` `OTC` `Slashdot` `Epinions` are used for the sign prediction;
  - `network1` `network2` `network3` are used for the node classification, and `community1` `community2` `community3` contain the corresponding labels;
  - `network1` `network_N5k` `network_N10k` `network_N50k` `network_N100k` are used for run time test.
- `src/` contains the model and necessary functions.

## Requirements
The implementation is tested under Python 3.7, with the following packages installed:
- `networkx==2.3`
- `numpy==1.16.5`
- `texttable==1.6.2`
- `tqdm==4.36.1`

## Input
The code takes a graph in `.txt` format as input. Every row of the data file indicates an edge between two nodes separated by `space` or `\t`. Nodes id should starts from any non-negative number. An example structure of the input file is the following:

| Source node | Target node | Sign |
| :-----:| :----: | :----: |
| 0 | 1 | -1 |
| 1 | 3 | 1 |
| 1 | 2 | 1 |
| 2 | 4 | -1 |

**NOTE** All the used graphs are **directed**. However, if you want to handle an **undirected** graph, modify your input file to make that each edge (u, v, s) constitutes two rows of the file:

| Source node | Target node | Sign |
| :-----:| :----: | :----: |
| u | v | s |
| v | u | s |

## Options
```
--dataset                   STR       Name of the input file
--comm-path                 STR       Path of the file containing community labels
--epoch-num                 INT       Number of training epochs
--dim                       INT       Dimension of representations
--k                         INT       Number of noise samples per data sample
--h                         INT       Highest order         
--test-size                 FLOAT     Test ratio
--split-seed                INT       Random seed for splitting dataset
--sign-prediction           BOOL      Whether conduct sign prediction
--node-classification       BOOL      Whether conduct node classification
```
**NOTE** When training the model, we used the entire network for node classification, but the partial network for sign prediction. Thus, `sign-prediction` and `node-classification` cannot be Ture, simultaneously. If you are conducting sign prediction, just ignore the parameter `comm-path`
