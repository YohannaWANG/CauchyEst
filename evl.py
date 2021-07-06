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

import math


def nCr(n,k):
    '''
    Function for calculate combination
    C(n, k) = n!/(k! * (n-k)!), for 0 <= k <= n
    '''
    f = math.factorial
    return f(n) // f(k) // f(n-k)

def DCP_rs(data, A_true, A_est, M_gt, Z):
    from scipy import stats
    '''
    Performance evaluation (D_cp distance over all samples)
    Based on this paper:(DCP calculation for noisy data through robust statistics - Not used in our paper)
        https://feb.kuleuven.be/public/u0017833/PDF-FILES/Croux_Dehon5.pdf
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

        ''' Calculate a_true and a_est'''

        index_true = A_true[:, child]
        index_est = A_est[:, child]
        
        a_true = index_true[index_true != 0]
        a_est = index_est[index_est != 0]
        

        ''' delta = [a_true - a_est]'''

        delta = a_true - a_est

        ''' Calculate sigma_y (true)'''

        if len(a_est) == 0:
            points = data[:, child]
            a = np.abs(data[:, child])
            
        elif len(a_est) == 1:      
            points = data[:, child] - a_est * np.transpose(data[:, parents])
            points = np.squeeze(points)
            a = np.abs(data[:, child] - a_est * np.transpose(data[:, parents]))

        elif len(a_est) > 1:
            points = data[:, child] - np.matmul(np.array(a_est), np.transpose(data[:, parents]))       
            a = np.abs(data[:, child] - np.matmul(np.array(a_est), np.transpose(data[:, parents])))
        
        dist = []        
        for i in list(range(len(points))):
            for j in list(range(i)): 
                d = np.abs(points[i] - points[j])
                dist.append(d)

        dist_new = np.sort(dist)
        index = nCr(np.round(len(points)/2) + 1, 2)
        smallest_dist = dist_new[index]
        sigma_hat =  2.219 *smallest_dist    
        
        sigma_hat2 = np.median(a)/0.674
        

        sigma_y = np.sqrt(np.diag(Z)[child])


        if sigma_y == 0 or sigma_hat == 0:
            DCP = 0
            
        else:
            ''' DCP can be calculated as follows: '''


            if len(delta) == 1:
                DMD = (delta * M * delta)/(2 * np.square(sigma_hat))

            else:
                DMD = np.matmul(np.matmul(np.transpose(
                                delta), M), delta)/(2 * np.square(sigma_hat))
            
        
            dcp = np.log(sigma_hat/sigma_y) + (np.square(sigma_y) - np.square(sigma_hat))/(2*np.square(sigma_hat)) + DMD

            DCP = np.append(DCP, dcp)

            
    KL = np.sum(DCP)

    return KL

    
def DCP2(data, A_true, A_est, M_gt, Z):

    '''
    Performance evaluation (D_cp distance over all samples)
    NOTES: Used only in our noisy data experiments.

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


        ''' Calculate a_true and a_est'''

        index_true = A_true[:, child]
        index_est = A_est[:, child]
        
        a_true = index_true[index_true != 0]
        a_est = index_est[index_est != 0]
        

        ''' delta = [a_true - a_est]'''

        delta = a_true - a_est

        ''' Calculate sigma_y (true)'''

        if len(a_est) == 0:
            a = np.abs(data[:, child])
            trim_med = np.median(a)/0.674
            sigma_hat = np.sqrt(trim_med)
            
        elif len(a_est) == 1:
            a = np.abs(data[:, child] - a_est * np.transpose(data[:, parents]))
            trim_med = np.median(a)/0.674
            sigma_hat = np.sqrt(trim_med)

        elif len(a_est) > 1:
            a = np.abs(data[:, child] - np.matmul(np.array(a_est), np.transpose(data[:, parents])))
            trim_med = np.median(a)/0.674
            sigma_hat = np.sqrt(trim_med)
            

        sigma_y = np.sqrt(np.diag(Z)[child])

        if sigma_y == 0 or sigma_hat == 0:
            DCP = 0
            
        else:
            ''' DCP can be calculated as follows: '''


            if len(delta) == 1:
                DMD = (delta * M * delta)/(2 * np.square(sigma_hat))

            else:
                DMD = np.matmul(np.matmul(np.transpose(
                                delta), M), delta)/(2 * np.square(sigma_hat))
            dcp = np.log(sigma_hat/sigma_y) + (np.square(sigma_y) - np.square(sigma_hat))/(2*np.square(sigma_hat)) + DMD

            DCP = np.append(DCP, dcp)

            
    KL = np.sum(DCP)

    return KL



"""
Performance evaluation:
    KL-Distance VS sample size
"""
    
def DCP1(data, A_true, A_est, M_gt, Z):
    '''
    Performance evaluation (D_cp distance over all samples)
    NOTES: Used in all other experiments (except noisy data exp)

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


        if sigma_y == 0 or sigma_hat == 0:
            DCP = 0
            
        else:
            ''' DCP can be calculated as follows: '''


            if len(delta) == 1:
                DMD = (delta * M * delta)/(2 * np.square(sigma_hat))

            else:
                DMD = np.matmul(np.matmul(np.transpose(
                                delta), M), delta)/(2 * np.square(sigma_hat))

            dcp = np.log(sigma_hat/sigma_y) + (np.square(sigma_y) - np.square(sigma_hat))/(2*np.square(sigma_hat)) + DMD

            DCP = np.append(DCP, dcp)

    KL = np.sum(DCP)

    return KL


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

        ''' Calculate a_true and a_est'''

        index_true = A_true[:, child]
        index_est = A_est[:, child]

        a_true = index_true[index_true != 0]
        a_est = index_est[index_true != 0]
        

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


        if sigma_y == 0 or sigma_hat == 0:
            DCP = 0
            
        else:
            ''' DCP can be calculated as follows: '''


            if len(delta) == 1:
                DMD = (delta * M * delta)/(2 * np.square(sigma_hat))

            else:
                DMD = np.matmul(np.matmul(np.transpose(
                                delta), M), delta)/(2 * np.square(sigma_hat))

            dcp = np.log(sigma_hat/sigma_y) + (np.square(sigma_y) - np.square(sigma_hat))/(2*np.square(sigma_hat)) + DMD

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

    
 
    
