#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 17:17:57 2021

@author: yohanna
"""
import config
import numpy as np
import itertools
from data import SynDAG

p = config.setup()
lgr = p.logger



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

def empirical_est(data):
    '''
    The empirical estimator (like in Appendix C of https://arxiv.org/pdf/1710.05209.pdf);
        where cov = 1/m * E(X^T * X)
        
    Arguments:
        data    : Input test data;
    
    Return:   
        cov_est : Estimated empirical covariance matrix from testing data
    '''

    empir_cov_np = np.cov(data.T, bias=True)
    empir_cov = np.matmul(data.T, data)/p.s
    
    return empir_cov_np, empir_cov

if __name__ == '__main__':
    Input = SynDAG(p)
    #Input.visualise()
    W_DAG = Input.A
    B_DAG = Input.B
    data = Input.X 
    train_data, test_data = split_data(data)   
    
    dic_cov_idx, dic_cov_val, cov_gt = ground_truth_cov(train_data, B_DAG)    
    
    empir_cov_np, empir_cov = empirical_est(test_data)
    
    print('Empirical cov_numpy', empir_cov_np)
    print('empirical cov      ', empir_cov)
    
    print('Ground truth cov   ', cov_gt)
    