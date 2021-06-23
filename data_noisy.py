#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Yuhao WANG
Data:  2020-07-22
Function: Load data from "generate_data.py", then introduce missingness
"""
import numpy as np
import pandas as pd
import networkx as nx
import missingno as msno
import matplotlib.pyplot as plt

#import generate_data
#from utils import args


from data import SynDAG

import config
p = config.setup()
lgr = p.logger



"""
    Function: This is the missingness mechanism introduced by "structure learning 
    under missing data". They firstly use 'noisy_or' function to assign probability
    fo the variables which would have missing entries. Then use the multiplied 
    probability as the ground truth.
"""


def missing_noisy_or(x1, x2):
    thresh = 1
    p1 = 0.7
    p2 = 0.75
    pL = 0.92
    or1 = x1 > thresh
    or2 = x2 > thresh
    n = len(x1)
    prob = np.ones((n, 1))
    for i in range(n):
        if or1[i] * or2[i] == 1:
            prob[i] = 1 - (1 - p1) * (1 - p2)
        elif or1[i] == 1 and or2[i] == 0:
            prob[i] = p1
        elif or1[i] == 0 and or2[i] == 1:
            prob[i] = p2
        else:
            prob[i] = pL
    return prob


# ------------------------------------------------------------------------------
"""
    Function: generate MAR missingness based on the noisy-or missingness mechanism
              This is to generate the baseline results based on the paper "
              Structure learning under missing data"
"""


def determin_miss_entry(Graph):
    """
    Function: list the parent-children sets of a 'Graph' G.
              Usage: The parent-children pairs will be further used to generate
                      Missing-AT-Random (MAR) data for causal discovery.
    """
    pairs = list(Graph.edges())

    parents = np.array([])
    children = np.array([])
    for parent, child in pairs:
        parents = np.append(parents, parent)
        children = np.append(children, child)
    """
    Function: Then randomly select the parents node, which will lead 
              to the missingness of their children node.
    Return: the selected parents-children pairs
    """
    parents_uniq = np.unique(parents)
    missing_parents = np.random.choice(parents_uniq, int(np.floor(len(parents_uniq))))

    missing_sets = np.array([])
    for i in missing_parents:
        for j, k in pairs:
            if i == j:
                miss_pair = np.array((j, k))
                missing_sets = np.append(missing_sets, miss_pair, axis=0)
    missing_sets = missing_sets.reshape(-1, 2)
    _, indices = np.unique(missing_sets[:, 0], return_index=True)
    missing_sets_wd = missing_sets[indices, :]
    print("missing sets without duplicate(wd): ", missing_sets_wd)
    if len(missing_sets_wd[:, 1]) == len(np.unique(missing_sets_wd[:, 1])):
        print('Nodes selection is correct, please continue!')
    else:
        print('Child node selection has duplicated values. Do Re-selection')
        _, indices_c = np.unique(missing_sets_wd[:, 1], return_index=True)
        missing_sets_wd = missing_sets_wd[indices_c, :]

    return np.array(missing_sets_wd)


# ------------------------------------------------------------------------------
def mar_missing_data_ipw(n, d, data, Graph, m_threshold=0.1):
    """
    Function: Given n(row_num), d(col_num), data, Graph
    Return  : missing_set_wd --- parent-child pair index for MAR missingness;
              p --- The ground truth probability for IPW baseline (Inverse Probability Weighting);
              data_incomplete --- Data contains Nan missing;
              data_remove(Xcom) --- List-wise deletion of missing entries, resulting
                                    in a complete observation;
              missing_mask --- 0/1 missing mask indicate missingness in data_incomplete
    """

    missing_sets_wd = determin_miss_entry(Graph)
    """
    Function: initialize R, and generate P (Inverse Probability)
    """
    R = np.ones((n, d))

    for parent, child in missing_sets_wd:
        R[:, int(child)] = np.squeeze(missing_noisy_or(data.iloc[:, int(parent)], data.iloc[:, int(child)]), axis=1)

    P = np.array([])
    for i in range(n):  # R.shape[0]
        prod = np.prod(R[i, :]) ** (-1)
        P = np.append(P, prod)
    P = np.expand_dims(P, axis=1)

    """
    Function: Get a copy from X_true into data_incomplete, and introduce missingness in data_incomplete
              based on R (data_incomplete). Then generate W_true (Inverse Probability Weighting).
    """
    data_incomplete = data.copy()
    R3 = R.copy()

    """Determine the missingness threshold"""
    W_true = []
    for parent, child in missing_sets_wd:  # missing_sets_wd[:,0]: #1
        child = int(child)
        R_tmp = np.expand_dims(R3[:, child], axis=1)
        missing_mask = np.random.rand(*R_tmp.shape) > m_threshold
        R_tmp = missing_mask * 1
        for j in range(n):  # R.shape[0]
            # R_tmp[j] = np.random.choice((0,1), 1, p=[float(1-R[j,child]), float(R[j,child])])
            if R_tmp[j] == 0:
                data_incomplete.iloc[j, int(parent)] = None
        R3[:, child] = np.squeeze(R_tmp, axis=1)

    for i in range(n):
        if np.prod(R3[i, :]) == 1:
            W_true.append(float(P[i]))

    """
    Function: Then perform list-wise deletion and generate the complete observation
              from Xt:a That is Xt -> Xcom.
    """
    data_remove = data_incomplete[~np.isnan(data_incomplete).any(axis=1)]

    missing_mask_Xt = data_incomplete.notnull().astype('int')

    return missing_sets_wd, R3, W_true, missing_mask_Xt, data_incomplete, data_remove


# ------------------------------------------------------------------------------
"""
    Function: Generate missing data by python 
"""


def missing_method(raw_data):
    """ Make the random value predictiable"""
    # np.random.seed(0)
    data_incomplete = raw_data.copy()
    m_threshold = p.th
    rows, cols = data_incomplete.shape

    if p.mechanism == 'mcar':
        if p.method == 'uniform':
            """
            Function: Missing Completely At Random
            """
            missing_mask = np.random.rand(*raw_data.shape) < m_threshold
            data_incomplete[missing_mask] = np.nan
            nan = np.count_nonzero(np.isnan(data_incomplete))
            print('nan = ', nan)
            print("Missing_mcar_proportion = " + "{:.2%}".format(np.true_divide(nan, data_incomplete.size)))

        elif p.method == 'random':
            """
            Function: Missing At Random. Missing pattern is caused by fully
                     observed variables
            """
            # Half attributes have missing valyes
            missing_cols = np.random.choice(cols, cols // 2, replace=False)
            c = np.zeros(cols, dtype=bool)
            c[missing_cols] = True
            missing_mask = (np.random.rand(*data_incomplete.shape) < m_threshold) * c
            data_incomplete[missing_mask] = np.nan
            nan = np.count_nonzero(np.isnan(data_incomplete))
            print("Missing_mcar_proportion = " + "{:.2%}".format(np.true_divide(nan, data_incomplete.size)))
        else:
            raise Exception('Method: No such method, please check parser variable.')

    if p.mechanism == 'mnar':
        if p.method == 'uniform':
            """ Can't control missing proportion"""
            sample_cols = np.random.choice(cols, 2)
            m1, m2 = np.median(data_incomplete[:, sample_cols], axis=0)
            m1 = data_incomplete[:, sample_cols[0]] <= m1
            m2 = data_incomplete[:, sample_cols[1]] >= m2
            m = (m1 * m2)[:, np.newaxis]
            missing_mask = (np.random.rand(*data_incomplete.shape) < m_threshold) * m
            data_incomplete[missing_mask] = np.nan
            nan = np.count_nonzero(np.isnan(data_incomplete))
            print("Missing_mcar_proportion = " + "{:.2%}".format(np.true_divide(nan, data_incomplete.size)))

        elif p.method == 'random':
            """
            Function: Missing Not at Random (only half of the attributes have
                                             missing values.)
            """
            missing_cols = np.random.choice(cols, cols // 2)
            c = np.zeros(cols, dtype=bool)
            c[missing_cols] = True

            sample_cols = np.random.choice(cols, 2)
            m1, m2 = np.median(data_incomplete[:, sample_cols], axis=0)
            v = np.random.uniform(size=(rows, cols))
            # missing values where (v<=t) and (x1 <= m1 or x2 >= m2)
            m1 = data_incomplete[:, sample_cols[0]] <= m1
            m2 = data_incomplete[:, sample_cols[1]] >= m2
            m = (m1 * m2)[:, np.newaxis]

            missing_mask = m * (v <= m_threshold) * c
            data_incomplete[missing_mask] = np.nan
            nan = np.count_nonzero(np.isnan(data_incomplete))
            print("Missing_mcar_proportion = " + "{:.2%}".format(np.true_divide(nan, data_incomplete.size)))

        else:
            raise Exception('Method: No such method, please check parser variable.')
            # else:
    #    raise Exception('Mechanism: No such method, please check parser variable.')


    mis_data = pd.DataFrame(data_incomplete)
    msno.matrix(mis_data.sample(data_incomplete.shape[0]))
    plt.show()
    msno.bar(mis_data.sample(data_incomplete.shape[0]))
    plt.show()
    new = ~np.array(missing_mask)

    return new * 1, data_incomplete




if __name__ == "__main__":

    p.s = 100
    Input = SynDAG(p)
    data = Input.X
    Graph =  Input.B

    missing_mask, data_incomplete = missing_method(data)



    #data_incomplete = pd.DataFrame(data_incomplete, columns=["V{}".format(i) for i in range(d)])

    # print(data_incomplete)