#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 11:57:04 2021

@author: yohanna wang
@email: yohanna.wang0924@gmail.com

"""
import os
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import config

p = config.setup()
lgr = p.logger

"""
Function: (Data) Generate synthetic DAG data (linear)
"""


class SynDAG:
    '''class for synthetically-generated directed acyclic graph (DAG) data'''

    def __init__(self, p):
        self.p = p

        B = SynDAG.DAG(p)
        A = SynDAG.WDAG(B, p)
        self.X = SynDAG.SEM(A, p)

        self.A, self.B = A, B

    @staticmethod
    def ER(p):
        '''
        simulate Erdos-Renyi (ER) DAG through networkx package

        Arguments:
            p: argparse arguments

        Uses:
            p.n: number of nodes
            p.d: degree of graph
            p.rs: numpy.random.RandomState

        Returns:
            B: (n, n) numpy.ndarray binary adjacency of ER DAG
        '''
        n, d, s = p.n, p.d, p.rs
        p = float(d)/(n-1)

        G = nx.generators.erdos_renyi_graph(n=n, p=p, seed=s)
        U = nx.to_numpy_matrix(G)
        B = np.tril(U, k=-1)

        return B

    @staticmethod
    def RT(p):
        '''
        simulate Random Tree DAG through networkx package

        Arguments:
        Arguments:
            p: argparse arguments

        Uses:
            p.n: number of nodes
            p.rs: numpy.random.RandomState
        '''
        n, s = p.n, p.s
        G = nx.random_tree(n, seed=s)
        U = nx.to_numpy_matrix(G)
        B = np.tril(U, k=-1)

        return B

    @staticmethod
    def DAG(p):
        '''
        simulate a directed acyclic graph (DAG)

        Arguments:
            p: argparse arguments

        Uses:
            p.n: number of nodes
            p.d: degree of graph
            p.tg: type of graph
            p.rs: numpy.random.RandomState

        Returns:
            B: (n, n) numpy.ndarray binary adjacency of DAG (permuted).
        '''
        t = p.tg.lower()
        if t == 'er':
            B = SynDAG.ER(p)
        elif t == 'rt':
            B = SynDAG.RT(p)
        else:
            raise ValueError(
                "The type of graph is either unknown or has not been implemented")

        # np.random.permutation permutes first axis only
        P = p.rs.permutation(np.eye(p.n))
        B = P.T @ B @ P

        return B

    @staticmethod
    def WDAG(B, p):
        '''
        simulate a weighted directed acyclic graph (DAG)

        Arguments:
            B: binary adjacency of DAG
            p: argparse arguments

        Uses:
            p.sf: scaling factor for range of weights for DAG
            p.rs: numpy.random.RandomState

        Returns:
            A: (n, n) numpy.ndarray weighted adjacency matrix of DAG.
        '''
        A = np.zeros(B.shape)

        s = p.sf
        R = ((-2*s, -0.5*s), (0.5*s, 2*s))
        S = p.rs.randint(len(R), size=A.shape)

        for i, (l, h) in enumerate(R):
            U = p.rs.uniform(low=l, high=h, size=A.shape)
            A += B * (S == i) * U

        return A

    @staticmethod
    def SEM(A, p):
        '''
        simulate samples from linear structural equation model (SEM) with specified type of noise.

        Arguments:
            A: (n, n) weighted adjacency matrix of DAG
            p: argparse arguments

        Uses:
            p.n: number of nodes
            p.s: number of samples
            p.rs (numpy.random.RandomState): Random number generator
            p.tn: type of noise, options: ev, nv, exp, gum
                ev: equal variance
                uv: unequal variance
                exp: exponential noise
                gum: gumbel noise

        Returns:
            numpy.ndarray: (s, n) data matrix.
        '''
        s, r, t = p.s, p.rs, p.tn.lower()

        def _SEM(X, I):
            '''
            simulate samples from linear SEM for the i-th vertex

            Arguments:
                X (numpy.ndarray): (s, number of parents of vertex i) data matrix
                I (numpy.ndarray): (n, 1) weighted adjacency vector for the i-th node

            Returns:
                numpy.ndarray: (s, 1) data matrix.
            '''
            if t == 'ev':
                N = r.normal(scale=1.0, size=s)
            elif t == 'uv':
                N = r.normal(scale=r.uniform(low=1.0, high=2.0), size=s)
            elif t == 'exp':
                N = r.exponential(scale=1.0, size=s)
            elif t == 'gum':
                N = r.gumbel(scale=1.0, size=s)
            else:
                raise ValueError('unknown noise type')

            return X @ I + N

        n = p.n
        X = np.zeros([s, n])
        G = nx.DiGraph(A)

        for v in list(nx.topological_sort(G)):
            P = list(G.predecessors(v))
            X[:, v] = _SEM(X[:, P], A[P, v])

        return X

    def visualise(self, D=[], U=[], name='true'):
        '''
        Arguments:
            D: numpy adjacency of directed edges
            U: list of undirected edges
        '''
        if D == []:
            D = self.A
        n, log = D.shape[1], self.p.log

        def _edges(G, s):
            D = nx.get_edge_attributes(G, s)
            if s != 'weight':
                return [D[k] for k in D.keys()]
            return D

        G = nx.DiGraph(D)
        E = _edges(G, 'weight')
        for (u, v) in E.keys():
            G.add_edge(u, v, width=2, weight=round(E[(u, v)], 2))

        pos = nx.nx_pydot.graphviz_layout(G)
        nx.draw(G, pos=pos, with_labels=True, font_size=10)
        nx.draw_networkx_nodes(G, pos=pos, node_size=500)

        nx.draw_networkx_edges(G, pos=pos, width=_edges(G, 'width'))
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=_edges(G, 'weight'), font_size=6)
        if log:
            plt.savefig(os.path.join(self.p.dir, name+'.png'), dpi=1000)


"""
Function: (Data) Load real BN data (from bnlearn);
TODO: Debug this part 
"""


def load_real_data():
    """
    Function: Load real 
    """
    from rpy2.robjects.packages import importr
    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects.vectors import StrVector
    from rpy2.robjects.packages import STAP

    import rpy2.robjects.numpy2ri
    from rpy2.robjects import pandas2ri

    #import os
    #os.environ['R_HOME'] = '/home/yohanna/R/x86_64-pc-linux-gnu-library/4.0'

    pandas2ri.activate()
    rpy2.robjects.numpy2ri.activate()

    packageNames = ('graph', 'RBGL', 'ggm', 'mgcv', 'pcalg')
    utils = rpackages.importr('utils')

    with open('utils.R', 'r') as f:
        string = f.read()

    bayesian_network = STAP(string, "chain")

    """
    Choices including: 
    1. Gaussian Bayesian networks
      Medium networks(20-50 nodes):
      > ECOLI70: (46 nodes, 70 arcs, 162 parameters)
      > MEGIC-NIAB: (44 nodes, 66 arcs, 154 parameters)
      Details: https://www.bnlearn.com/bnrepository/gaussian-medium.html#magic-niab
      
      Large networks(50-100 nodes):
      > MAGIC-IRRI: (64 nodes, 102 arcs, 230 parameters)
      Details: https://www.bnlearn.com/bnrepository/gaussian-large.html#magic-irri
      
      Very large networks(101-1000 nodes):
      > ARTH150: (107 nodes, 150 arcs, 364 parameters)
      Details: https://www.bnlearn.com/bnrepository/gaussian-verylarge.html#arth150
    
    2. Conditional Linear Gaussian Bayesian Networks
      Small networks(<20 nodes):
      > HEALTHCARE (7 nodes, 9 arcs, 42 parameters)
      > SANGIOVESE (15 nodes, 55 arcs, 259 parameters)
      Details: https://www.bnlearn.com/bnrepository/clgaussian-small.html#healthcare
      
      Medium networks(20-50 nodes)
      > MEHRA (24 nodes, 71 arcs, 324423 parameters)
      Details: https://www.bnlearn.com/bnrepository/clgaussian-medium.html#mehra
    """
    choice = p.choice
    graph_sets = bayesian_network.get_graph(choice)

    G_binary = graph_sets[0]
    G_weighted = graph_sets[1]
    interception = graph_sets[2]

    """
    TODO: since bnlearn package only offers a graph structure;
    We need to further generate observational data based on the graph
    """
    data = 1
    return G_binary, G_weighted  # data


"""
TODO: (Baseline Algorithm) For the experiments, we need to compare to 
(I)  The empirical estimator (like in Appendix C of https://arxiv.org/pdf/1710.05209.pdf);
(II) Glasso/clime (algorithms for undirected graphical models). 
(III) Compare to the algorithm that solves least squares at each node.
      This is the max likelihood estimator for gaussian.
Notes: This algorithm will use the structure of the Bayes net, 
       and I expect it will need similar number of samples as our algorithm
"""
def clime(data):
    '''
    learn cov_est use CLIME algorithm.
    Paper: http://stat.wharton.upenn.edu/~tcai/paper/Precision-Matrix.pdf
    R package: https://cran.r-project.org/web/packages/flare/index.html
    
    Arguments:
        data    : Input test data;
    
    Return:   
        cov_est : Estimated empirical covariance matrix from testing data
    '''
    # TODO: Add clime algorithm
    print('ADD CLIME ALGORITHM')
    
def empirical_est(data):
    '''
    The empirical estimator (like in Appendix C of https://arxiv.org/pdf/1710.05209.pdf);
        where cov = 1/m * E(X^T * X)
        
    Arguments:
        data    : Input test data;
    
    Return:   
        cov_est : Estimated empirical covariance matrix from testing data
    '''
    # TODO: Q    np.cov(data.T)
    return (np.matmul(data.T, data)/p.s)


def glasso(data):

    from sklearn.covariance import GraphicalLasso

    '''
    Graphical Lasso algorithm from sklearn package
    
    Arguments:
        data    : Input data;
    
    Return:   
        cov_est : Graph with learned coefficients
        
    '''
    cov = GraphicalLasso().fit(data)
    cov_est = np.around(cov.covariance_, decimals=3)
    
    return cov_est

def DKL_ud(cov_est, cov_gt):
    '''
    Calculate KL-distance for undirected graph (different from DCP<directed graph>)
        Link: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    
    Arguments:
        cov_est : estimated covariance matrix from (1) empirical estimator; (2) GLASSO;
        
        cov_gt  : ground truth covariance matrix;
        
    Return:
        DKL_ud  : KL-distance for undirected graph
    '''

    dkl = 1/2*np.trace(np.matmul(np.linalg.inv(cov_est), cov_gt)) - cov_gt.shape[0]\
        + np.log((np.linalg.det(cov_est)/(np.linalg.det(cov_gt))))

    return dkl
    

def eval_un(train_data, test_data, A_bin):
    '''
    Evaluate the KL disctance between undirected graph using the D_KL equation 
     for multivarite normal distribution:

    Arguments:
        data    : Input data;
    
    Return:   
        cov_est : Graph with learned coefficients   
    '''
    _, _, cov_gt = ground_truth_cov(train_data, A_bin)
    cov_est = glasso(test_data)
    
    dkl = DKL_ud(cov_est, cov_gt)
    return dkl   
    
    
def regression(data, A_bin):

    from sklearn.linear_model import LinearRegression
    '''
    Learn coefficients through linear regression
    
    Arguments:
        data    : Input data;
        A_bin   : The graph structure 
        
    Return:
        A_est   : Graph with learned coefficients
    '''
    n = p.n
    A_est = np.zeros((n, n))

    for child in range(n):
        parents = [list(pa) for pa in (np.nonzero(A_bin[:, child]))]
        parents = list(itertools.chain(*parents))

        if len(parents) > 0:
            for pa in parents:
                data_ch = data[:, child].reshape(-1, 1)
                data_pa = data[:, pa].reshape(-1, 1)
                '''
                Notes: Linear regression returns 'Coefficient of determination',
                       which is R-square.
                       
                       Coefficient of correlation is “R” value which is given 
                       in the summary table in the Regression output. 
                '''
                reg = LinearRegression().fit(data_ch, data_pa)

                A_est[pa, child] = float(reg.coef_)

    return A_est


def sigma_estimator(data, A_est):
    '''
    Algorithm 1: Recovering the varianceσgiven an estimatêAof coefficientsA
    '''
    n = p.n
    Sigma_hat = {}
    for child in range(n):
        parents = [list(pa) for pa in (np.nonzero(A_est[:, child]))]
        parents = list(itertools.chain(*parents))

        ''' Calculate a_est'''

        index_est = A_est[:, child]
        a_est = index_est[index_est != 0]

        ''' Calculate sigma_y (true)'''

        if len(a_est) == 0:
            sigma_hat = np.var(data[:, child])

        elif len(a_est) == 1:
            sigma_hat = np.var(data[:, child] - a_est *
                               np.transpose(data[:, parents]))

        elif len(a_est) > 1:
            sigma_hat = np.var(
                data[:, child] - np.matmul(np.array(a_est), np.transpose(data[:, parents])))
        
        #print('sigma_hat', sigma_hat)
        Sigma_hat[child] = sigma_hat
    
    return Sigma_hat


def least_square(data, A_bin):
    '''
    Algorithm 2:
        Recovery algorithm for general Bayesian networks via Least Squares estimators

    Learn coefficients through linear regression

    Arguments:
        data    : Input data;
        A_bin   : The graph structure 

    Return:
        A_est   : Graph with learned coefficients

    '''
    n = p.n
    A_est = np.zeros((n, n))

    for child in range(n):
        parents = [list(pa) for pa in (np.nonzero(A_bin[:, child]))]
        parents = list(itertools.chain(*parents))

        if len(parents) > 0:
            for pa in parents:

                X = data[:, pa]
                Y = data[:, child]
                '''
                Notes: A_hat = (X^T*X)^{-1}*X^T*Y
                '''
                a_est = np.matmul(np.matmul(np.expand_dims((1/np.matmul(np.transpose(X), X)),
                                                           axis=0), np.expand_dims(np.transpose(X), axis=0)), Y)
                A_est[pa, child] = a_est

    return A_est


def median_estimator(data, A_bin):
    '''
    Algorithm 3: 
        Recovery algorithm for tree-skeletoned Bayesian networks

    Learn coefficients use median

    Arguments:
        data    : Input data;
        A_bin   : The graph structure 

    Return:
        A_est   : Graph with learned coefficients
    '''
    n, d = p.n, p.d
    A_est = np.zeros((n, n))

    for child in range(n):
        parents = [list(pa) for pa in (np.nonzero(A_bin[:, child]))]
        parents = list(itertools.chain(*parents))

        if len(parents) > 0:
            '''
            P: number of parents, 1 <= p <= d
            '''
            P = len(parents)

            A_est_s = []  
            for s in range(data.shape[0] - d + 1):
                X = data[s:s+P, parents]
                Y = np.expand_dims(data[s:s+P, child], axis=1)
                
                a_est_s = np.matmul(np.linalg.inv(X), Y)
                A_est_s = np.append(A_est_s, a_est_s)
            
            ''' Find the median '''                
            A_est_s = A_est_s.reshape(-1, P).T
            A_est_median = np.median(A_est_s, axis=1)

            for i in range(len(parents)):
                A_est[parents[i], child] = A_est_median[i]

    return A_est
                
def heuristic_extension(data, A_bin):
    
    import scipy.linalg
    '''
    Algorithm 4: Recovery algorithm for general Bayesian networks.
    
    Estimate L_hat using empirical covariance matrix M_hat.
    Learn coefficients from median
    
    Arguments:
        data    : Input data;
        A_bin   : The graph structure 

    Return:
        A_est   : Graph with learned coefficients
    '''
    
    n, d = p.n, p.d
    A_est = np.zeros((n, n))
    for child in range(n):
        parents = [list(pa) for pa in (np.nonzero(A_bin[:, child]))]
        parents = list(itertools.chain(*parents))


        ''' Calculate M: covariance matrix among parents'''
        if len(parents) > 0:
            if len(parents) == 1:
                M = np.var(data[:, parents].T)
            else:
                M = np.cov(data[:, parents].T)
     
            
            ''' Compute Cholesky decomposition M_hat = L_hat * L_hat^T'''
            L = scipy.linalg.cholesky(M, lower=True)
            '''
            p: number of parents, 1 <= p <= d
            '''
            P = len(parents)

            A_est_s = []  
            
            for s in range(data.shape[0] - d + 1):
                X = data[s:s+P, parents]
                Y = np.expand_dims(data[s:s+P, child], axis=1)     
                
                a_est_s = np.matmul(np.linalg.inv(X), Y)
                A_est_s = np.append(A_est_s, a_est_s)
            
            ''' Find the median '''                
            A_est_s = A_est_s.reshape(-1, P).T
            MED = np.median(np.matmul(np.transpose(L), A_est_s), axis=1)
            
            ''' Define estimates A_hat '''
            A_est_i = np.matmul(np.linalg.inv(np.transpose(L)), np.transpose(MED))
            
            for i in range(len(parents)):
                A_est[parents[i], child] = A_est_i[i]
                
            
    return A_est



"""
Performance evaluation:
    KL-Distance VS sample size
"""
    
def DCP(data, A_true, A_est):
    '''
    Performance evaluation (D_cp distance over all samples)

    Arguments:
        data    : Input data;
        A_true  : hidden true parameters A_true = (A, sigma_y);
        A_est   : our estimates          A_est  = (A, sigma_hat);

    Returns:
        DKL = sum(DCP) KL divergence. 
    '''
    n = p.n
    DCP = np.array([])
    for child in range(n):
        parents = [list(pa) for pa in (np.nonzero(A_true[:, child]))]
        parents = list(itertools.chain(*parents))

        ''' Calculate M: covariance matrix among parents'''

        if len(parents) == 1:
            M = np.var(data[:, parents].T)
        else:
            M = np.cov(data[:, parents].T)

        child_data = data[:, child]
        parents_data = data[:, parents]

        ''' Calculate a_true and a_est'''

        index_true = A_true[:, child]
        index_est = A_est[:, child]
        a_true = index_true[index_true != 0]
        a_est = index_est[index_est != 0]

        ''' delta = [a_true - a_est]'''

        delta = a_true - a_est

        ''' Calculate sigma_y (true)'''

        if len(a_est) == 0:
            sigma_hat = np.var(data[:, child])
            sigma_y = np.var(data[:, child])

        elif len(a_est) == 1:
            sigma_hat = np.var(data[:, child] - a_est *
                               np.transpose(data[:, parents]))
            sigma_y = np.var(data[:, child] - a_true *
                             np.transpose(data[:, parents]))

        elif len(a_est) > 1:
            sigma_hat = np.var(
                data[:, child] - np.matmul(np.array(a_est), np.transpose(data[:, parents])))
            sigma_y = np.var(
                data[:, child] - np.matmul(np.array(a_true), np.transpose(data[:, parents])))

        ''' DCP can be calculated as follows: '''
        if len(delta) == 1:
            DMD = (delta * M * delta)/(2 * np.square(sigma_hat))

        else:
            DMD = np.matmul(np.matmul(delta, M), np.transpose(
                delta))/(2 * np.square(sigma_hat))

        dcp = np.log(sigma_hat/sigma_y) + (np.square(sigma_y) -
                                           np.square(sigma_hat))/(2*np.square(sigma_hat)) + DMD

        DCP = np.append(DCP, dcp)

    return np.sum(DCP)


def split_data(data):
    '''
    Split the data into training (eg. 80%) and testing (eg. 20%)
    Arguments:
        data    : Input data;

    Returns:
        train_data: training data
        test_data : testing data
    
    '''
    train_prop, test_prop = p.train, p.test 
    train_index = int(train_prop * data.shape[0])
    
    train_data = data[0:train_index, :]
    test_data = data[train_index:, :]
    
    return train_data, test_data
    


def ground_truth_cov(data, A_bin):
    '''
    Given training data, get the ground truth covariance matrix between each
        [child, parents] data, and the covariance matrix over whole training data.
        Therefore, if there are n nods, we will return n+1 matrix.
        
     Arguments:
        data    : Input data;
        A_bin   : Binary adjacency matrix;

    Returns:
        dic_cov : a dictionary stores all the ground truth covariance matrix.
       
    '''
    n = p.n
    dic_cov_idx = {}
    dic_cov_val = {}
    
    for child in range(n):
        parents = [list(pa) for pa in (np.nonzero(A_bin[:, child]))]
        parents = list(itertools.chain(*parents))

        '''
        Notes: For each child node, M is the covairnce matrix between its parents. 
        '''
        if len(parents) == 0:
            M = 0
        elif len(parents) == 1:
            M = np.var(data[:, parents].T)
        elif len(parents) > 1:
            M = np.cov(data[:, parents].T)
        else:
            raise ValueError('unknown parents size')
            
        dic_cov_idx[child] = parents
        dic_cov_val[child] = M
        
    gt_cov = np.cov(data.T)
    
    return dic_cov_idx, dic_cov_val, gt_cov
     

def my_code():
    '''
    Run our algorithm
    '''
    Input = SynDAG(p)
    Input.visualise()
    W_DAG = Input.A
    B_DAG = Input.B
    data = Input.X 
    
    train_data, test_data = split_data(data)   
    
    dic_cov_idx, dic_cov_val, cov_gt = ground_truth_cov(train_data, B_DAG)    
    
    #median_estimator(data, B_DAG)
    A_est = heuristic_extension(data, B_DAG)
    sigma_estimator(data, A_est)
    
    """
    ''' For directed graph (first two) and undirected graph (glasso, empirical)'''
    if p.algorithm == 'regression':
        A_est = regression(test_data, B_DAG)
        
    elif p.algorithm == 'least_square':
        A_est = least_square(data, B_DAG)

    elif p.algorithm == 'glasso':
        cov_est = glasso(test_data)
        dkl = DKL_ud(cov_est, cov_gt)
    
    elif p.algorithm == 'empirical':
        cov_est = empirical_est(test_data)
        dkl = DKL_ud(cov_est, cov_gt)
    else:
        raise ValueError('unknown algorithm')    
    #eval_un(train_data,test_data, B_DAG)


    #kl_reg = DCP(data, W_DAG, A_est)

    print('A_est', A_est)
    print('W dag = ', W_DAG)
    #print('KL distance', kl_reg)

    """
if __name__ == '__main__':

    my_code()
