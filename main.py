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
from evl import DKL_ud, eval_un, DCP

p = config.setup()
lgr = p.logger



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


def batch_least_square(data, A_bin):
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
    mb_size = p.batch
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
            #for pa in parents:

            X = data[:, parents]
            Y = data[:, child]
            '''
            Notes: A_hat = (X^T*X)^{-1}*X^T*Y
            '''
            #a_est1 = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), Y)
            a_est = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)

            
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
    

def my_code_ud():
    
    Input = SynDAG(p)
    Input.visualise()
    W_DAG = Input.A

    B_DAG = Input.B
    data = Input.X 
    Z = Input.Z
    
    I = np.identity(p.n)
    A = np.linalg.inv(I - W_DAG)
    B = np.transpose(np.linalg.inv(I - W_DAG))
    M = np.matmul(np.matmul(A, Z), B)

    train_data, test_data = split_data(data)   
    
    dic_cov_idx, dic_cov_val, dic_cov_val_gt, cov_gt = ground_truth_cov(train_data, B_DAG, M)    

    

    ''' Undirected graph'''
    #print('GLASSO')
    #cov_glasso_est = glasso_R(test_data)
    print('EMP')
    cov_emp_np, cov_emp_est = empirical_est(test_data)
    #print('CLIME')
    #cov_clime_est  = clime(test_data)
    #cov_tiger_est  = tiger(test_data)
    
    #kl_glasso = DKL_ud(cov_glasso_est, cov_gt)    
    #kl_emp_np = DKL_ud(cov_emp_np, cov_gt)
    #kl_emp = DKL_ud(cov_emp_est, cov_gt)  
    #kl_clime = DKL_ud(cov_clime_est, cov_gt)
    #kl_tiger = DKL_ud(cov_tiger_est, cov_gt)
    
    
    ''' Directed graph'''
    #A_est_reg = regression(test_data, B_DAG)
    print('LS')
    A_est_ls  = least_square(test_data, B_DAG)
    print('Batch LS')
    A_est_ls_batch = batch_least_square(data, B_DAG)
    print('CAUest')
    A_est_cau_med = CauchyEst_median(data, B_DAG)
    print('HECAU')
    A_est_he_med = heuristic_extension_median(data, B_DAG)
    #A_est_cau_trimmed = CauchyEst_trimmed(test_data, B_DAG)
    #A_est_he_trimmed = heuristic_extension_trimmed(data, B_DAG)
    
    
    #sigma_reg = sigma_estimator(data, A_est_ls)
    
    #kl_reg = DCP(train_data, test_data, W_DAG, A_est_reg)
    kl_ls  = DCP(train_data, test_data,  W_DAG, A_est_ls, M, Z) 
    kl_ls_batch  = DCP(train_data, test_data,  W_DAG, A_est_ls, M, Z)
    kl_cau_med = DCP(train_data, test_data,  W_DAG, A_est_cau_med, M, Z)
    kl_he_med  = DCP(train_data, test_data,  W_DAG, A_est_he_med, M, Z)
    #kl_cau_tri = DCP(train_data, test_data,  W_DAG, A_est_cau_trimmed)
    #kl_he_tri  = DCP(train_data, test_data,  W_DAG, A_est_he_trimmed)
    
    #return kl_glasso, kl_emp_np, kl_emp, kl_clime, kl_tiger, kl_ls, kl_cau_med, kl_he_med
    return  kl_ls, kl_ls_batch, kl_cau_med, kl_he_med


def test():
    KL_LS = []
    KL_HE_MED = []
    
    for i in range(10):
        Input = SynDAG(p)
        Input.visualise()
        W_DAG = Input.A
        print('W_DAG', W_DAG)
        B_DAG = Input.B
        data = Input.X 
        Z = Input.Z  # Z is sigma^2
        
        I = np.identity(p.n)
        A = np.linalg.inv(I - W_DAG)
        B = np.linalg.inv(np.transpose(I - W_DAG))
        ''' '''        
        M = np.matmul(np.matmul(A, Z), B)

        
        train_data, test_data = split_data(data)   
        
        dic_cov_idx, dic_cov_val, dic_cov_val_gt, cov_gt = ground_truth_cov(train_data, B_DAG, M)   
        #cov_emp_np, cov_emp_est = empirical_est(test_data)
        #kl_emp = DKL_ud(cov_emp_np, cov_gt)
        #A_est_ls  = least_square(data, B_DAG)
        A_est_he_med = heuristic_extension_median(data, B_DAG)
        
        #A_est_ls = batch_least_square(data, B_DAG)
        #kl_ls  = DCP(train_data, test_data,  W_DAG, A_est_ls, M, Z)
        
        kl_he_med  = DCP(train_data, test_data,  W_DAG, A_est_he_med, M, Z)
        
        #KL_LS.append(kl_ls)
        KL_HE_MED.append(kl_he_med)
    
    #print('KL_LS', KL_LS)
    #print('LS_mean ', np.mean(KL_LS))
    #print('LS_std ', np.std(KL_LS))

    print('KL_HE_MED', KL_HE_MED)
    print('KL_HE_mean ', np.mean(KL_HE_MED))
    print('KL_HE_std ', np.std(KL_HE_MED))
    
    
def main():
    print('node number is ', p.n)
    KL_LS  = []
    KL_LS_BATCH = []
    KL_CAU_MED = []
    KL_HE_MED = []
    
    KL_CAU_TRI = []
    KL_HE_TRI  = []
    
    KL_GLASSO = []
    KL_EMP_NP = []
    KL_EMP    = []
    KL_CLIME  = []
    KL_TIGER  = []
    

    for i in range(6):
        print('i = ', i)
 
        #kl_glasso,  kl_emp_np, kl_emp, kl_clime, kl_tiger, kl_ls, kl_cau_med, kl_he_med = my_code_ud()
        #kl_emp_np, kl_emp, kl_clime, kl_ls, kl_ls_batch, kl_cau_med, kl_he_med = my_code_ud()
        kl_ls, kl_ls_batch, kl_cau_med, kl_he_med = my_code_ud()
        #KL_GLASSO.append(kl_glasso)
        #KL_EMP_NP.append(kl_emp_np)
        #KL_EMP.append(kl_emp)
        #KL_CLIME.append(kl_clime)
        #KL_TIGER.append(kl_tiger)

        KL_LS.append(kl_ls)      
        KL_LS_BATCH.append(kl_ls_batch)
        KL_CAU_MED.append(kl_cau_med)
        KL_HE_MED.append(kl_he_med)
        #KL_CAU_TRI.append(kl_cau_tri)
        #KL_HE_TRI.append(kl_he_tri)
    
    
    #print('KL_GLASSO = ', np.mean(KL_GLASSO))
    #print('KL_EMP_NP = ', np.mean(KL_EMP_NP))
    #print('KL_EMP = ', np.mean(KL_EMP))
    #print('KL_CLIME = ', np.mean(KL_CLIME))
    #print('KL_TIGER = ', np.mean(KL_TIGER))

    print('KL_LS  =', np.mean(KL_LS))
    print('KL_LS_BATCH = ', np.mean(KL_LS_BATCH))
    print('KL_MED = ', np.mean(KL_CAU_MED))  # cauchy median
    print('KL_HE = ', np.mean(KL_HE_MED))
   
    #print('KL_CAU_TRI =', np.mean(KL_CAU_TRI))
    #print('KL_HE_TRI  =', np.mean(KL_HE_TRI))
    
    #print('KL_GLASSO_std = ', np.std(KL_GLASSO))
    #print('KL_EMP_NP = ', np.std(KL_EMP_NP))
    #print('KL_EMP_std = ', np.std(KL_EMP))
    #print('KL_CLIME_std = ', np.std(KL_CLIME))
    #print('KL_TIGER_std = ', np.std(KL_TIGER))

    print('KL_LS_std  =', np.std(KL_LS))
    print('KL_LS_BATCH_std = ', np.std(KL_LS_BATCH))
    print('KL_MED_std = ', np.std(KL_CAU_MED))  # cauchy median
    print('KL_HE_std = ', np.std(KL_HE_MED))

if __name__ == '__main__':
    
    #main()
    test()
    #my_code_ud()











    
