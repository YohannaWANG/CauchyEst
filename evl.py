#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 09:06:52 2021

@author: yohanna
"""
import itertools
import numpy as np

import config
p = config.setup()

"""
Performance evaluation:
    KL-Distance VS sample size
"""
    
def DCP(data, A_true, A_est, M_gt, Z):
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
            
        M = M_gt[np.ix_(parents, parents)]
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
            sigma_hat = np.sqrt(np.mean(np.square(data[:, child])))
            
        elif len(a_est) == 1:
            sigma_hat = np.sqrt(np.mean(np.square(data[:, child] - a_est *
                               np.transpose(data[:, parents]))))

        elif len(a_est) > 1:
            sigma_hat = np.sqrt(np.mean(np.square(
                data[:, child] - np.matmul(np.array(a_est), np.transpose(data[:, parents])))))
            

        sigma_y = np.sqrt(np.diag(Z)[child])

        sigma_y = 1
        sigma_hat = 1
        if sigma_y == 0 or sigma_hat == 0:
            DCP = 0
        else:
            ''' DCP can be calculated as follows: '''


            if len(delta) == 1:
                DMD = (delta * M * delta)/(2 * np.square(sigma_hat))

            else:
                DMD = np.matmul(np.matmul(np.transpose(
                                delta), M), delta)/(2 * np.square(sigma_hat))
            dcp = np.log(sigma_hat/sigma_y) + (np.square(sigma_y) -
                                               np.square(sigma_hat))/(2*np.square(sigma_hat)) + DMD

            DCP = np.append(DCP, dcp)

    KL = np.sum(DCP)

    return KL



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

    
 
    