#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 11:57:04 2021

@author: yohanna wang
@email: yohanna.wang0924@gmail.com

"""
import os
import time
import itertools
import numpy as np
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import config

from data import SynDAG

p = config.setup()
lgr = p.logger



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
    print('glasso_cov', glasso_cov)
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
    print('np cov is ', np.cov(data.T))
    empir_cov_np = np.cov(data.T, bias=True)
    empir_cov = np.matmul(data.T, data)/p.s
    print('empir_cov is ', empir_cov)
    
    return empir_cov_np, empir_cov

@ignore_warnings(category=ConvergenceWarning)
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

    dkl = 1/2*(np.trace(np.matmul(np.linalg.inv(cov_est), cov_gt)) - cov_gt.shape[0]\
        + np.log((np.linalg.det(cov_est)/(np.linalg.det(cov_gt)))))

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
    cov_est = glasso(test_data, tol=0.01, max_iter=1000, normalize=True)
    
    dkl = DKL_ud(cov_est, cov_gt)
    return dkl   
    
    

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


def CauchyEst_median(data, A_bin):
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

def heuristic_extension_median(data, A_bin):
    
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
    
def DCP(train_data, test_data, A_true, A_est):
    '''
    Performance evaluation (D_cp distance over all samples)

    Arguments:
        data    : Input data;
                {train_data, test_data}
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
            M = np.var(test_data[:, parents].T)
        else:
            M = np.cov(test_data[:, parents].T)

        #child_data = data[:, child]
        #parents_data = data[:, parents] for each nodeuniform variance varied from (1,2) 

        ''' Calculate a_true and a_est'''

        index_true = A_true[:, child]
        index_est = A_est[:, child]
        
        a_true = index_true[index_true != 0]
        a_est = index_est[index_est != 0]

        ''' delta = [a_true - a_est]'''

        delta = a_true - a_est

        ''' Calculate sigma_y (true)'''

        if len(a_est) == 0:
            sigma_hat = np.var(test_data[:, child])
            
            sigma_y = np.var(train_data[:, child])

        elif len(a_est) == 1:
            sigma_hat = np.var(test_data[:, child] - a_est *
                               np.transpose(test_data[:, parents]))
            
            sigma_y = np.var(train_data[:, child] - a_true *
                             np.transpose(train_data[:, parents]))

        elif len(a_est) > 1:
            sigma_hat = np.var(
                test_data[:, child] - np.matmul(np.array(a_est), np.transpose(test_data[:, parents])))
            
            sigma_y = np.var(
                train_data[:, child] - np.matmul(np.array(a_true), np.transpose(train_data[:, parents])))

        ''' DCP can be calculated as follows: '''
        #print('sigma_hat ', sigma_hat)
        
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
    #Input.visualise()
    W_DAG = Input.A
    B_DAG = Input.B
    data = Input.X 
    
    train_data, test_data = split_data(data)   
    
    dic_cov_idx, dic_cov_val, cov_gt = ground_truth_cov(train_data, B_DAG)    
    
    
    ''' Run our algorithm '''
    #A_est_reg = regression(test_data, B_DAG)
    A_est_ls  = least_square(test_data, B_DAG)
    A_est_cau_med = CauchyEst_median(data, B_DAG)
    A_est_he_med = heuristic_extension_median(data, B_DAG)
    #A_est_cau_trimmed = CauchyEst_trimmed(test_data, B_DAG)
    #A_est_he_trimmed = heuristic_extension_trimmed(data, B_DAG)
    
    
    #sigma_reg = sigma_estimator(data, A_est_ls)
    
    #kl_reg = DCP(train_data, test_data, W_DAG, A_est_reg)
    kl_ls  = DCP(train_data, test_data,  W_DAG, A_est_ls)
    kl_cau_med = DCP(train_data, test_data,  W_DAG, A_est_cau_med)
    kl_he_med  = DCP(train_data, test_data,  W_DAG, A_est_he_med)
    #kl_cau_tri = DCP(train_data, test_data,  W_DAG, A_est_cau_trimmed)
    #kl_he_tri  = DCP(train_data, test_data,  W_DAG, A_est_he_trimmed)
    

    return kl_ls, kl_cau_med, kl_he_med  # kl_cau_tri, kl_he_tri,
    

def my_code_ud():
    
    Input = SynDAG(p)
    #Input.visualise()
    W_DAG = Input.A
    B_DAG = Input.B
    data = Input.X 
    
    train_data, test_data = split_data(data)   
    
    dic_cov_idx, dic_cov_val, cov_gt = ground_truth_cov(train_data, B_DAG)    
    
    ''' Undirected graph'''
    cov_glasso_est = glasso_R(test_data)
    cov_emp_np, cov_emp_est = empirical_est(test_data)
    cov_clime_est  = clime(test_data)
    cov_tiger_est  = tiger(test_data)
    
    kl_glasso = DKL_ud(cov_glasso_est, cov_gt)    
    kl_emp_np = DKL_ud(cov_emp_np, cov_gt)
    kl_emp = DKL_ud(cov_emp_est, cov_gt)  
    kl_clime = DKL_ud(cov_clime_est, cov_gt)
    kl_tiger = DKL_ud(cov_tiger_est, cov_gt)
    
    
    ''' Directed graph'''
    #A_est_reg = regression(test_data, B_DAG)
    A_est_ls  = least_square(test_data, B_DAG)
    A_est_cau_med = CauchyEst_median(data, B_DAG)
    A_est_he_med = heuristic_extension_median(data, B_DAG)
    #A_est_cau_trimmed = CauchyEst_trimmed(test_data, B_DAG)
    #A_est_he_trimmed = heuristic_extension_trimmed(data, B_DAG)
    
    
    #sigma_reg = sigma_estimator(data, A_est_ls)
    
    #kl_reg = DCP(train_data, test_data, W_DAG, A_est_reg)
    kl_ls  = DCP(train_data, test_data,  W_DAG, A_est_ls)
    kl_cau_med = DCP(train_data, test_data,  W_DAG, A_est_cau_med)
    kl_he_med  = DCP(train_data, test_data,  W_DAG, A_est_he_med)
    #kl_cau_tri = DCP(train_data, test_data,  W_DAG, A_est_cau_trimmed)
    #kl_he_tri  = DCP(train_data, test_data,  W_DAG, A_est_he_trimmed)
    
    return kl_glasso, kl_emp_np, kl_emp, kl_clime, kl_tiger, kl_ls, kl_cau_med, kl_he_med

def main():
        
    KL_LS  = []
    KL_CAU_MED = []
    KL_HE_MED = []
    
    KL_CAU_TRI = []
    KL_HE_TRI  = []
    
    KL_GLASSO = []
    KL_EMP_NP = []
    KL_EMP    = []
    KL_CLIME  = []
    KL_TIGER  = []
    
    '''    
    for i in range(3):
        kl_glasso, kl_emp = my_code_ud()
        
        KL_GLASSO.append(kl_glasso)
        KL_EMP.append(kl_emp)
    
    print('KL_GLASSO = ', np.mean(KL_GLASSO))
    print('KL_EMP = ', np.mean(KL_EMP))
    
    '''
    for i in range(1):
        print('i = ', i)
 
        kl_glasso,  kl_emp_np, kl_emp, kl_clime, kl_tiger, kl_ls, kl_cau_med, kl_he_med = my_code_ud()
        
        KL_GLASSO.append(kl_glasso)
        KL_EMP_NP.append(kl_emp_np)
        KL_EMP.append(kl_emp)
        KL_CLIME.append(kl_clime)
        KL_TIGER.append(kl_tiger)
        
        KL_LS.append(kl_ls)        
        KL_CAU_MED.append(kl_cau_med)
        KL_HE_MED.append(kl_he_med)
        #KL_CAU_TRI.append(kl_cau_tri)
        #KL_HE_TRI.append(kl_he_tri)
    
    
    print('KL_GLASSO = ', np.mean(KL_GLASSO))
    print('KL_EMP_NP = ', np.mean(KL_EMP_NP))
    print('KL_EMP = ', np.mean(KL_EMP))
    print('KL_CLIME = ', np.mean(KL_CLIME))
    print('KL_TIGER = ', np.mean(KL_TIGER))

    print('KL_LS  =', np.mean(KL_LS))
    print('KL_MED = ', np.mean(KL_CAU_MED))  # cauchy median
    print('KL_HE = ', np.mean(KL_HE_MED))
   
    #print('KL_CAU_TRI =', np.mean(KL_CAU_TRI))
    #print('KL_HE_TRI  =', np.mean(KL_HE_TRI))
    


if __name__ == '__main__':
    
    main()












    
