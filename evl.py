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
    
def DCP(train_data, test_data, A_true, A_est, M_gt, Z):
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
    DCP1 = np.array([])
    DCP2 = np.array([])
    
    for child in range(n):
        parents = [list(pa) for pa in (np.nonzero(A_true[:, child]))]
        parents = list(itertools.chain(*parents))

        ''' Calculate M: covariance matrix among parents'''

        if len(parents) == 1:
            M = np.var(train_data[:, parents].T)
        else:
            M = np.cov(train_data[:, parents].T)
            
        M2 = M_gt[np.ix_(parents, parents)]
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
            sigma_hat = np.sqrt(np.mean(np.square(test_data[:, child])))
            
            #sigma_y_2 = np.sqrt(np.var(test_data[:, child]))
            #print('sigma_y_2', sigma_y_2)

        elif len(a_est) == 1:
            sigma_hat = np.sqrt(np.mean(np.square(test_data[:, child] - a_est *
                               np.transpose(test_data[:, parents]))))
            
            #sigma_y = np.var(train_data[:, child] - a_true *
            #                np.transpose(train_data[:, parents]))

        elif len(a_est) > 1:
            sigma_hat = np.sqrt(np.mean(np.square(
                test_data[:, child] - np.matmul(np.array(a_est), np.transpose(test_data[:, parents])))))
            
            #sigma_y = np.var(
            #    train_data[:, child] - np.matmul(np.array(a_true), np.transpose(train_data[:, parents])))
        sigma_y = np.sqrt(np.diag(Z)[child])

        #sigma_hat = np.sqrt(sigma_hat)
        #print('sigma_hat after square', sigma_hat)
        ''' DCP can be calculated as follows: '''
        #print('sigma_hat ', sigma_hat)
        
        if len(delta) == 1:
            DMD = (delta * M * delta)/(2 * np.square(sigma_hat))

        else:
            DMD = np.matmul(np.matmul(np.transpose(
                            delta), M), delta)/(2 * np.square(sigma_hat))
            
        dcp1 = np.log(sigma_hat/sigma_y) + (np.square(sigma_y) -
                                           np.square(sigma_hat))/(2*np.square(sigma_hat)) + DMD
        
        if len(delta) == 1:
            DMD2 = (delta * M2 * delta)/(2 * np.square(sigma_hat))

        else:
            DMD2 = np.matmul(np.matmul(np.transpose(
                            delta), M), delta)/(2 * np.square(sigma_hat))       
            
        dcp2 = np.log(sigma_hat/sigma_y) + (np.square(sigma_y) -
                                           np.square(sigma_hat))/(2*np.square(sigma_hat)) + DMD2


        DCP1 = np.append(DCP1, dcp1)
        DCP2 = np.append(DCP2, dcp2)
    KL1 = np.sum(DCP1)
    KL2 = np.sum(DCP2)

    return KL2#np.sum(DCP1)



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
    