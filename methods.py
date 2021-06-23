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

    
    return empir_cov_np, empir_cov

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
                #print('parents bigger than 1')
                M = np.cov(data[:, parents].T)
    

            ''' Compute Cholesky decomposition M_hat = L_hat * L_hat^T'''

            #M2 = np.array(M, dtype=np.float128)
            #print('M eigval', np.linalg.eigvals(M2))
            #M = np.array(M, dtype=np.float128)

            
            
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
            
            
            #L = cholesky(M)
  
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
    percent_of_sample = 5
    num_noise_node = 2
    sample_size = int(s * percent_of_sample / 100)

    sample = np.sort(np.random.choice(s, size=sample_size, replace=False))

    for i in sample:
        nodes = np.sort(np.random.choice(n, size=num_noise_node, replace=False))
        data[i, nodes] = np.random.normal(mu, sigma, num_noise_node)#np.array(10000 * np.ones(n))

    return data


def data_uniform(data):

    from data_noisy import missing_method
    missing_mask, data_incomplete = missing_method(data)

def main():
    sample = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

    KL_LS = []
    KL_BATCH_MED = []
    KL_BATCH_MEAN = []
    KL_CAU_TREE = []
    KL_CAU_GEN = []

    KL_LS_STD = []
    KL_BATCH_MED_STD = []
    KL_BATCH_MEAN_STD = []
    KL_CAU_TREE_STD = []
    KL_CAU_GEN_STD = []

    for i in range(len(sample)):
        print('sample ', i)
        for j in range(10):
            print('j', j)
            kl_ls = []
            kl_batch_med = []
            kl_batch_mean = []
            kl_cau_tree = []
            kl_cau_gen = []

            p.s = sample[i]
            Input = SynDAG(p)
            # Input.visualise()
            # plt.show()
            W_DAG = Input.A

            B_DAG = Input.B

            """Notes: prune a degree 10 graph into degree 5 graph"""
            # B_DAG = prune_graph(B_DAG_gt)
            # W_DAG = W_DAG_gt * B_DAG

            data = Input.X
            # data = nodes_to_big_value(data)
            data = list_to_big_value(data)
            #print(data)

            Z = Input.Z
            I = np.identity(p.n)
            A = np.linalg.inv(I - W_DAG)
            B = np.transpose(np.linalg.inv(I - W_DAG))
            M = np.matmul(np.matmul(A, Z), B)

            A_est_ls = least_square(data, B_DAG)
            kl1 = DCP(data, W_DAG, A_est_ls, M, Z)
            kl_ls.append(kl1)
            # print('kl1 \n', kl1)

            A_est_ls_batch_med_20 = batch_least_square_median(data, B_DAG, 20)
            kl2 = DCP(data, W_DAG, A_est_ls_batch_med_20, M, Z)
            kl_batch_med.append(kl2)
            # print('kl2 \n', kl2)

            A_est_ls_batch_mean_20 = batch_least_square_mean(data, B_DAG, 20)
            kl3 = DCP(data, W_DAG, A_est_ls_batch_mean_20, M, Z)
            kl_batch_mean.append(kl3)
            # print('kl3 \n', kl3)

            A_est_cau_med = CauchyEst_median(data, B_DAG)
            kl4 = DCP(data, W_DAG, A_est_cau_med, M, Z)
            kl_cau_tree.append(kl4)
            # print('kl4 \n', kl4)

            A_est_he_med = heuristic_extension_median(data, B_DAG)
            kl5 = DCP(data, W_DAG, A_est_he_med, M, Z)
            kl_cau_gen.append(kl5)

            # print('kl5 \n', kl5)
        ''' Least square'''

        kl_ls = np.median(kl_ls)
        kl_ls_std = np.std(kl_ls)

        KL_LS.append(kl_ls)
        KL_LS_STD.append(kl_ls_std)

        ''' Least square batch median'''
        kl_batch_med = np.mean(kl_batch_med)
        kl_batch_med_std = np.std(kl_batch_med)

        KL_BATCH_MED.append(kl_batch_med)
        KL_BATCH_MED_STD.append(kl_batch_med_std)

        ''' Least square batch mean'''
        kl_batch_mean = np.mean(kl_batch_mean)
        kl_batch_mean_std = np.std(kl_batch_mean)

        KL_BATCH_MEAN.append(kl_batch_mean)
        KL_BATCH_MEAN_STD.append(kl_batch_mean_std)

        '''Cauchy Tree '''
        kl_cau_tree = np.mean(kl_cau_tree)
        kl_cau_tree_std = np.std(kl_cau_tree)

        KL_CAU_TREE.append(kl_cau_tree)
        KL_CAU_TREE_STD.append(kl_cau_tree_std)

        '''Cauchy General '''
        kl_cau_gen = np.mean(kl_cau_gen)
        kl_cau_gen_std = np.std(kl_cau_gen)

        KL_CAU_GEN.append(kl_cau_gen)
        KL_CAU_GEN_STD.append(kl_cau_gen_std)

    print('KL_LS', KL_LS)
    print('KL_BATCH_MED', KL_BATCH_MED)
    print('KL_BATCH_MEAN', KL_BATCH_MEAN)
    print('KL_CAU_TREE', KL_CAU_TREE)
    print('KL_CAU_GEN', KL_CAU_GEN)

    plt.figure(figsize=(18, 10))

    plt.errorbar(sample, KL_LS, yerr=KL_LS_STD, label='LS', linewidth=3)
    plt.errorbar(sample, KL_BATCH_MED, yerr=KL_BATCH_MED_STD, label='Batch_MED_LS (batch=20)', linewidth=3)
    plt.errorbar(sample, KL_BATCH_MEAN, yerr=KL_BATCH_MEAN_STD, label='Batch_MEAN_LS (batch=20)', linewidth=3)
    plt.errorbar(sample, KL_CAU_TREE, yerr=KL_CAU_TREE_STD, label='CauchyEstTree', linewidth=3)
    plt.errorbar(sample, KL_CAU_GEN, yerr=KL_CAU_GEN_STD, label='CauchyEstGeneral', linewidth=3)

    plt.legend(loc='upper right', fontsize=32)
    plt.tick_params(axis='both', which='major', labelsize=35)
    plt.xlabel('Sample size (100 nodes, 5% noisy sample, 2 noisy nodes) ', fontsize=35)
    plt.ylabel('KL_Divergence', fontsize=35)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(18, 10))
    plt.errorbar(sample, KL_CAU_TREE, yerr=KL_CAU_TREE_STD, label='CauchyEstTree', linewidth=5)
    plt.errorbar(sample, KL_CAU_GEN, yerr=KL_CAU_GEN_STD, label='CauchyEstGeneral', linewidth=5)

    plt.legend(loc='upper right', fontsize=32)
    plt.tick_params(axis='both', which='major', labelsize=35)
    plt.xlabel('Sample size (100 nodes, 5% noisy sample, 2 noisy nodes) ', fontsize=35)
    plt.ylabel('KL_Divergence', fontsize=35)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
