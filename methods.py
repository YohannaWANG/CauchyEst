#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 10:17:50 2021

@author: yohanna
"""
import itertools
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import config
p = config.setup()
lgr = p.logger



from data import SynDAG
from evl import DCP


def clime(data):
    '''
    learn cov_est use CLIME algorithm.
    Paper: http://stat.wharton.upenn.edu/~tcai/paper/Precision-Matrix.pdf
    R package: https://cran.r-project.org/web/packages/flare/index.html
    
    Arguments:
        data    : Input test data;
    
    Return:   
        clime_cov_est : Estimated empirical covariance matrix from testing data
    Notes:
        CLIME solves the following minimization problem
                    min ||Ω||_1 s.t. ||SΩ − I||∞ ≤ λ,
    '''

    import rpy2.robjects as robjects
    from rpy2.robjects.packages import STAP

    import rpy2.robjects.numpy2ri
    from rpy2.robjects import pandas2ri

    pandas2ri.activate()
    rpy2.robjects.numpy2ri.activate()
    robjects.r.source("utils.R")

    with open('utils.R', 'r') as f:
        string = f.read()

    bayesian_network = STAP(string, "bayesian_network")
    clime_cov_est = bayesian_network.clime(data)

    return clime_cov_est

def tiger(data):
    '''
    learn cov_est use TIGER algorithm.
    Paper: http://stat.wharton.upenn.edu/~tcai/paper/Precision-Matrix.pdf
    R package: https://cran.r-project.org/web/packages/flare/index.html
    
    Arguments:
        data    : Input test data;
    
    Return:   
        tiger_cov_est : Estimated empirical covariance matrix from testing data

    '''

    import rpy2.robjects as robjects
    from rpy2.robjects.packages import STAP

    import rpy2.robjects.numpy2ri
    from rpy2.robjects import pandas2ri

    pandas2ri.activate()
    rpy2.robjects.numpy2ri.activate()
    robjects.r.source("utils.R")

    with open('utils.R', 'r') as f:
        string = f.read()

    bayesian_network = STAP(string, "bayesian_network")
    tiger_cov = bayesian_network.tiger(data)

    return tiger_cov

def glasso_R(data):
    '''
    learn cov_est use TIGER algorithm.
    Paper: http://stat.wharton.upenn.edu/~tcai/paper/Precision-Matrix.pdf
    R package: https://cran.r-project.org/web/packages/flare/index.html
    
    Arguments:
        data    : Input test data;
    
    Return:   
        tiger_cov_est : Estimated empirical covariance matrix from testing data

    '''

    import rpy2.robjects as robjects
    from rpy2.robjects.packages import STAP

    import rpy2.robjects.numpy2ri
    from rpy2.robjects import pandas2ri

    pandas2ri.activate()
    rpy2.robjects.numpy2ri.activate()
    robjects.r.source("utils.R")

    with open('utils.R', 'r') as f:
        string = f.read()

    bayesian_network = STAP(string, "bayesian_network")
    glasso_cov = bayesian_network.glasso_r(data)

    return glasso_cov

def empirical_est(data):
    '''
    The empirical estimator (like in Appendix C of https://arxiv.org/pdf/1710.05209.pdf);
        where cov = 1/m * E(X^T * X)
        
    Arguments:
        data    : Input test data;
    
    Return:   
        cov_est : Estimated empirical covariance matrix from testing data
    '''
    #return  np.cov(data.T) #(np.matmul(data.T, data)/p.s)

    empir_cov_np = np.cov(data.T, bias=True)

    b = data - np.mean(data, axis=0)
    empir_cov = np.matmul(np.transpose(b), b)/data.shape[0]

    
    return empir_cov

#@ignore_warnings(category=ConvergenceWarning)
def glasso(data): 

    from sklearn.covariance import GraphicalLasso

    '''
    Graphical Lasso algorithm from sklearn package
    
    Arguments:
        data    : Input data;
    
    Return:   
        cov_est : Graph with learned coefficients
        
    '''
    cov = GraphicalLasso(mode='lars', max_iter=2000).fit(data)
    cov_est = np.around(cov.covariance_, decimals=3)
    
    return cov_est


def sigma_estimator(data, A_est):
    '''
    Algorithm 1: Recovering the varianceσgiven an estimatêA of coefficientsA
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
            sigma_hat = np.sqrt(np.var(data[:, child]))

        elif len(a_est) == 1:
            sigma_hat = np.sqrt(np.var(data[:, child] - a_est *
                               np.transpose(data[:, parents])))

        elif len(a_est) > 1:
            sigma_hat = np.sqrt(np.var(
                data[:, child] - np.matmul(np.array(a_est), np.transpose(data[:, parents]))))
        
        Sigma_hat[child] = sigma_hat
    
    return Sigma_hat


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


def batch_least_square_mean(data, A_bin, mb_size):
    '''
    Algorithm 6:
        Recovery algorithm for general Bayesian networks via Batch Least Squre + x

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
            a_est_list = []
            
            for i in range(0, data.shape[0], mb_size+len(parents)):
                
                if i + mb_size+len(parents) < data.shape[0]:
                    X = data[i:i+mb_size+len(parents), parents]
                    Y = data[i:i+mb_size+len(parents), child]
                    '''
                    Notes: A_hat = (X^T*X)^{-1}*X^T*Y
                    '''
                    a_est = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
                    a_est_list.append(a_est)
                    
                else: break
            a_est_mean = np.mean(a_est_list, axis=0)

            A_est[parents, child] = a_est_mean
                
    return A_est

def batch_least_square_median(data, A_bin, mb_size):
    '''
    Algorithm 6:
        Recovery algorithm for general Bayesian networks via Batch Least Squre + x

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
            a_est_list = []
            
            for i in range(0, data.shape[0], mb_size+len(parents)):
                
                if i + mb_size+len(parents) < data.shape[0]:
                    X = data[i:i+mb_size+len(parents), parents]
                    Y = data[i:i+mb_size+len(parents), child]
                    '''
                    Notes: A_hat = (X^T*X)^{-1}*X^T*Y
                    '''
                    a_est = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
                    a_est_list.append(a_est)
                    
                else: break
            a_est_median = np.median(a_est_list, axis=0)

            A_est[parents, child] = a_est_median
                
    return A_est


def least_square_ill(data, A_bin):
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
    from sklearn.linear_model import Ridge, RidgeCV
    n = p.n
    A_est = np.zeros((n, n))

    for child in range(n):
        parents = [list(pa) for pa in (np.nonzero(A_bin[:, child]))]
        parents = list(itertools.chain(*parents))

        if len(parents) > 0:

            X = data[:, parents]
            Y = data[:, child]
            Y = np.expand_dims(Y, axis=1)
            '''
            Notes: A_hat = (X^T*X)^{-1}*X^T*Y
            '''
            #a_est = Ridge(alpha=1e-20) #1e-7, 1e-5, 1e-3, 1e-1, 1
            #a_est.fit(X, Y)
            a_est = RidgeCV(alphas=[1e-10, 1e-7, 1e-3, 1e-1, 1]).fit(X, Y)

            A_est[parents, child] = a_est.coef_

    return A_est


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

            X = data[:, parents]
            Y = data[:, child]
            '''
            Notes: A_hat = (X^T*X)^{-1}*X^T*Y
            '''
            #a_est1 = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), Y)

            a_est = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
            #a = np.dot(X.T, X)
            #b = np.dot(X.T, Y)
            #a_est = np.linalg.solve(a, b)
            
            A_est[parents, child] = a_est
                
    return A_est

def CauchyEst_trimmed(data, A_bin):
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
            P = int(len(parents))
            
            A_est_s = []  
            

            for s in range(data.shape[0] - d):
                
                X = data[s:s+P, parents]
                Y = np.expand_dims(data[s:s+P, child], axis=1)
                
                if X.shape[0] == X.shape[1]:
                    
                    a_est_s = np.matmul(np.linalg.inv(X), Y)
                    A_est_s = np.append(A_est_s, a_est_s)
                else:
                    break
            
            ''' Find the median '''             
            A_est_s = A_est_s.reshape(-1, P).T

            #A_est_median = np.median(A_est_s, axis=1)
            A_est_median = []
            
            for i in range(A_est_s.shape[0]):
                a_est_median = stats.trim_mean(A_est_s[i,:], 0.38)
                A_est_median.append(a_est_median)

            for i in range(len(parents)):
                A_est[parents[i], child] = A_est_median[i]

    return A_est


def CauchyEst_Tree(data, A_bin):
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
            P = int(len(parents))
            
            A_est_s = []  
            

            for s in range(data.shape[0] - P):
                
                X = data[s:s+P, parents]
                Y = np.expand_dims(data[s:s+P, child], axis=1)
                
                if X.shape[0] == X.shape[1]:
                    
                    #a_est_s = np.matmul(np.linalg.inv(X), Y)
                    a_est_s = np.linalg.solve(X, Y)
                    # Try use np.solve
                    A_est_s = np.append(A_est_s, a_est_s)
                else:
                    break
            
            ''' Find the median '''  
            A_est_s = A_est_s.reshape(-1, P).T
            

            A_est_median = np.median(A_est_s, axis=1)

            for i in range(len(parents)):
                A_est[parents[i], child] = A_est_median[i]

    return A_est
                
def heuristic_extension_trimmed(data, A_bin):
    
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
                
                if X.shape[0] == X.shape[1]:
                    a_est_s = np.matmul(np.linalg.inv(X), Y)
                    A_est_s = np.append(A_est_s, a_est_s)
                else:
                    break
            
            ''' Find the median '''                
            A_est_s = A_est_s.reshape(-1, P).T

            #MED = np.median(np.matmul(np.transpose(L), A_est_s), axis=1)
            
            temp = np.matmul(np.transpose(L), A_est_s)
            MED = []
            for i in range(A_est_s.shape[0]):
                med = stats.trim_mean(temp[i,:], 0.15)
                MED.append(med)            
            
            ''' Define estimates A_hat '''
            A_est_i = np.matmul(np.linalg.inv(np.transpose(L)), np.transpose(MED))
            
            for i in range(len(parents)):
                A_est[parents[i], child] = A_est_i[i]
                    
    return A_est

def CauchyEst_General(data, A_bin):
    
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
                #print('parents bigger than 1')
                M = np.cov(data[:, parents].T)
    

            ''' Compute Cholesky decomposition M_hat = L_hat * L_hat^T'''


            def cholesky(A):
                """Performs a Cholesky decomposition of A, which must 
                be a symmetric and positive definite matrix. The function
                returns the lower variant triangular matrix, L."""
                n = len(A)
            
                # Create zero matrix for L
                L = [[0.0] * n for i in range(n)]
            
                # Perform the Cholesky decomposition
                for i in range(n):
                    for k in range(i+1):
                        tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))
                        
                        if (i == k): # Diagonal elements
                            # LaTeX: l_{kk} = \sqrt{ a_{kk} - \sum^{k-1}_{j=1} l^2_{kj}}
                            L[i][k] = np.sqrt(A[i][i] - tmp_sum)
                        else:
                            # LaTeX: l_{ik} = \frac{1}{l_{kk}} \left( a_{ik} - \sum^{k-1}_{j=1} l_{ij} l_{kj} \right)
                            L[i][k] = (1.0 / L[k][k] * (A[i][k] - tmp_sum))
                return L

            L = scipy.linalg.cholesky(M, lower=True)    

            '''
            p: number of parents, 1 <= p <= d
            '''
            P = len(parents)

            A_est_s = []  
            
            for s in range(data.shape[0] - P):
                X = data[s:s+P, parents]
                Y = np.expand_dims(data[s:s+P, child], axis=1)     
                
                if X.shape[0] == X.shape[1]:
                    a_est_s = np.matmul(np.linalg.inv(X), Y)
                    A_est_s = np.append(A_est_s, a_est_s)
                else:
                    break
            
            ''' Find the median '''                
            A_est_s = A_est_s.reshape(-1, P).T

            MED = np.median(np.matmul(np.transpose(L), A_est_s), axis=1)

            ''' Define estimates A_hat '''
            A_est_i = np.matmul(np.linalg.inv(np.transpose(L)), np.transpose(MED))

            for i in range(len(parents)):
                A_est[parents[i], child] = A_est_i[i]

                    
    return A_est



def split_data(data):
    '''
    Split the data into training (eg. 80%) and testing (eg. 20%)
    Arguments:
        data    : Input data;

    Returns:
        train_data: training data
        test_data : testing data
    
    '''
    train_prop = p.train
    
    train_index = int(train_prop * data.shape[0])
    
    train_data = data[0:train_index, :]
    test_data = data[train_index:, :]
    
    return train_data, test_data
    


def ground_truth_cov(data, A_bin, M_gt):
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
    
    dic_cov_val_gt = {}
    
    for child in range(n):
        parents = [list(pa) for pa in (np.nonzero(A_bin[:, child]))]
        parents = list(itertools.chain(*parents))

        '''
        Notes: For each child node, M is the covairnce matrix between its parents. 
        '''
        if len(parents) == 0:
            M = 0
            M2 = 0
        elif len(parents) == 1:
            M = np.var(data[:, parents].T)
            M2 = M_gt[np.ix_(parents, parents)]
        elif len(parents) > 1:
            M = np.cov(data[:, parents].T)
            M2 = M_gt[np.ix_(parents, parents)]

        else:
            raise ValueError('unknown parents size')
            
        dic_cov_idx[child] = parents
        dic_cov_val[child] = M
        dic_cov_val_gt[child] = M2
        
    gt_cov = np.cov(data.T)
    
    return dic_cov_idx, dic_cov_val, dic_cov_val_gt, gt_cov


def nodes_to_big_value(data):
    n, s = p.n, p.s
    mu, sigma = 10000, 1
    num_noise_node = 2

    for i in range(s):
        nodes = np.sort(np.random.choice(n, size=num_noise_node, replace=False))
        data[i, nodes] = np.random.normal(mu, sigma, len(nodes)) #np.array(10000 * np.ones(s)) #
    return data


def list_to_big_value(data):
    n, s = p.n, p.s
    mu, sigma = 10000, 1
    percent_of_sample = 0
    num_noise_node = 2
    sample_size = int(s * percent_of_sample / 100)

    sample = np.sort(np.random.choice(s, size=sample_size, replace=False))

    for i in sample:
        nodes = np.sort(np.random.choice(n, size=num_noise_node, replace=False))
        data[i, nodes] = np.random.normal(mu, sigma, num_noise_node)#np.array(10000 * np.ones(n))

    return data




