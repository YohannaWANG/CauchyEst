#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:34:34 2021

@author: yohanna
"""
import numpy as np

from data import SynDAG
from evl import DCP

import config
p = config.setup()

from methods import least_square, batch_least_square_median, batch_least_square_mean, \
                    CauchyEst_median, heuristic_extension_median
                    
def get_data():
    
    Input = SynDAG(p)
    #Input.visualise()
    W_DAG = Input.A
    B_DAG = Input.B
    '''

    """Notes: prune a degree 10 graph into degree 5 graph"""
    B_DAG = prune_graph_fixed(B_DAG_gt)
    W_DAG = W_DAG_gt * B_DAG
    
    _, _, _, shd, _ = count_accuracy(nx.DiGraph(W_DAG_gt), nx.DiGraph(W_DAG))
    print('shd is ', shd)
    '''
    data = Input.X 
    Z = Input.Z
    
    I = np.identity(p.n)
    A = np.linalg.inv(I - W_DAG)
    B = np.transpose(np.linalg.inv(I - W_DAG))
    M = np.matmul(np.matmul(A, Z), B)
    
    
    return data, B_DAG, W_DAG, M, Z
    
def least_square_test(data, B_DAG):
    
    A_est_ls  = least_square(data, B_DAG)
    
    return A_est_ls

def batch_least_square_median_test(data, B_DAG):
    
    A_est_ls_batch_med_20 = batch_least_square_median(data, B_DAG, 20)
    
    return A_est_ls_batch_med_20

def batch_least_square_mean_test(data, B_DAG):
    
    A_est_ls_batch_mean_20 = batch_least_square_mean(data, B_DAG, 20)
    
    return A_est_ls_batch_mean_20

def CauchyEst_median_test(data, B_DAG):
    A_est_cau_med = CauchyEst_median(data, B_DAG)
    
    return A_est_cau_med

def heuristic_extension_median_test(data, B_DAG):

    A_est_he_med = heuristic_extension_median(data, B_DAG)
    
    return A_est_he_med


def algorithm1(data, B_DAG):
    return least_square_test(data, B_DAG)

def algorithm2(test_data, B_DAG):
    return batch_least_square_median_test(data, B_DAG)

def algorithm3(data, B_DAG):
    return batch_least_square_mean_test(data, B_DAG)

def algorithm4(test_data, B_DAG):
    return CauchyEst_median_test(test_data, B_DAG)
    
def algorithm5(data, B_DAG):
    return heuristic_extension_median_test(data, B_DAG)


def numbers_to_algorithms(argument):
    switcher = {
        1: algorithm1(data, B_DAG),
        2: algorithm2(data, B_DAG),
        3: algorithm3(data, B_DAG),
        4: algorithm4(data, B_DAG),
        5: algorithm5(data, B_DAG)
    }

    func = switcher.get(argument, lambda: "Invalid algorithm")

    return func

def KL1(data, B_DAG, W_DAG, M, Z):
    
    A_est_ls  = least_square(data, B_DAG)
    kl1 = DCP(data, W_DAG, A_est_ls, M, Z) 
    
    return kl1   
    
def KL2(data, B_DAG, W_DAG, M, Z):
    
    A_est_ls_batch_med_20 = batch_least_square_median(data, B_DAG, 20)
    kl2 = DCP(data, W_DAG, A_est_ls_batch_med_20, M, Z) 

    return kl2
    
def KL3(data, B_DAG, W_DAG, M, Z):
    
    A_est_ls_batch_mean_20 = batch_least_square_mean(data, B_DAG, 20)
    kl3 = DCP(data, W_DAG, A_est_ls_batch_mean_20, M, Z) 

    return kl3

def KL4(data, B_DAG, W_DAG, M, Z):
    
    A_est_cau_med = CauchyEst_median(data, B_DAG)
    kl4 = DCP(data, W_DAG, A_est_cau_med, M, Z) 

    return kl4

def KL5(data, B_DAG, W_DAG, M, Z):  
    
    A_est_he_med = heuristic_extension_median(data, B_DAG)
    kl5 = DCP(data, W_DAG, A_est_he_med, M, Z) 

    return kl5
    
def numbers_to_KL_divergence(argument):
    switcher = {
        1: KL1(data, B_DAG, W_DAG, M, Z),
        2: KL2(data, B_DAG, W_DAG, M, Z),
        3: KL3(data, B_DAG, W_DAG, M, Z),
        4: KL4(data, B_DAG, W_DAG, M, Z),
        5: KL5(data, B_DAG, W_DAG, M, Z)
    }

    func = switcher.get(argument, lambda: "Invalid algorithm")

    return func


def my_code_ud():

    Input = SynDAG(p)
    #Input.visualise()
    W_DAG = Input.A

    B_DAG = Input.B

    """Notes: prune a degree 10 graph into degree 5 graph"""
    #B_DAG = prune_graph(B_DAG_gt)
    #W_DAG = W_DAG_gt * B_DAG
    
    data = Input.X 
    Z = Input.Z
    
    I = np.identity(p.n)
    A = np.linalg.inv(I - W_DAG)
    B = np.transpose(np.linalg.inv(I - W_DAG))
    M = np.matmul(np.matmul(A, Z), B)

    #from methods import split_data, ground_truth_cov
    #train_data, test_data = split_data(data)   
    
    #dic_cov_idx, dic_cov_val, dic_cov_val_gt, cov_gt = ground_truth_cov(train_data, B_DAG, M)    



    ''' Undirected graph'''
    #print('GLASSO')
    #cov_glasso_est = glasso_R(test_data)
    #print('EMP')
    #cov_emp_np, cov_emp_est = empirical_est(test_data)
    #print('CLIME')
    #cov_clime_est  = clime(test_data)
    #cov_tiger_est  = tiger(test_data)
    
    #kl_glasso = DKL_ud(cov_glasso_est, cov_gt)    
    #kl_emp_np = DKL_ud(cov_emp_np, cov_gt)
    #kl_emp = DKL_ud(cov_emp_est, cov_gt)  
    #kl_clime = DKL_ud(cov_clime_est, cov_gt)
    #kl_tiger = DKL_ud(cov_tiger_est, cov_gt)
    
    
    ''' Directed graph'''

    print('LS')
    A_est_ls  = least_square(data, B_DAG)
    print('Batch LS median')
    A_est_ls_batch_median = batch_least_square_median(data, B_DAG, 20)
    print('Batch LS median')
    A_est_ls_batch_mean = batch_least_square_mean(data, B_DAG, 20)
    print('CAUest')
    A_est_cau_med = CauchyEst_median(data, B_DAG)
    print('HECAU')
    A_est_he_med = heuristic_extension_median(data, B_DAG)
    #A_est_cau_trimmed = CauchyEst_trimmed(test_data, B_DAG)
    #A_est_he_trimmed = heuristic_extension_trimmed(data, B_DAG)
    
    
    #sigma_reg = sigma_estimator(data, A_est_ls)
    
    #kl_reg = DCP(train_data, test_data, W_DAG, A_est_reg)
    kl_ls  = DCP(data,  W_DAG, A_est_ls, M, Z) 
    kl_ls_batch_median  = DCP(data,  W_DAG, A_est_ls_batch_median, M, Z)
    kl_ls_batch_mean  = DCP(data,  W_DAG, A_est_ls_batch_mean, M, Z)
    kl_cau_med = DCP(data,  W_DAG, A_est_cau_med, M, Z)
    kl_he_med  = DCP(data,  W_DAG, A_est_he_med, M, Z)
    
    print('ls ', kl_ls)
    print('batch_median', kl_ls_batch_median)
    print('batch_mean', kl_ls_batch_mean)
    print('cau', kl_cau_med)
    print('he ', kl_he_med)
    #kl_cau_tri = DCP(train_data, test_data,  W_DAG, A_est_cau_trimmed)
    #kl_he_tri  = DCP(train_data, test_data,  W_DAG, A_est_he_trimmed)
    

 
if __name__ == '__main__':
    
    Input = SynDAG(p)
    #Input.visualise()
    W_DAG = Input.A

    B_DAG = Input.B

    """Notes: prune a degree 10 graph into degree 5 graph"""
    #B_DAG = prune_graph(B_DAG_gt)
    #W_DAG = W_DAG_gt * B_DAG
    
    data = Input.X 
    Z = Input.Z
    
    I = np.identity(p.n)
    A = np.linalg.inv(I - W_DAG)
    B = np.transpose(np.linalg.inv(I - W_DAG))
    M = np.matmul(np.matmul(A, Z), B)

    #from methods import split_data, ground_truth_cov
    #train_data, test_data = split_data(data)   
    
    #dic_cov_idx, dic_cov_val, dic_cov_val_gt, cov_gt = ground_truth_cov(train_data, B_DAG, M)   
    
    #train_data, test_data, B_DAG, W_DAG, M, Z = get_data()
    
    
    print('Least Square \n', numbers_to_KL_divergence(1))
    print('Batch_least_square_median \n', numbers_to_KL_divergence(2))
    print('Batch_least_square_mean \n',numbers_to_KL_divergence(3))
    print('Cauchy Tree \n',numbers_to_KL_divergence(4))
    print('Cauchy General \n',numbers_to_KL_divergence(5))


    my_code_ud()
    '''
    
    print('=========================')
    print('data shape ', data.shape)
    A_est_ls  = least_square(data, B_DAG)
    kl1 = DCP(data, W_DAG, A_est_ls, M, Z) 
    print('KL 1 ', kl1)
    
    print('=========================')
    print('KL1 ', KL1(data, B_DAG, W_DAG, M, Z))
    print('KL2 ', KL2(data, B_DAG, W_DAG, M, Z))
    print('KL3 ', KL3(data, B_DAG, W_DAG, M, Z))
    print('KL4 ', KL4(data, B_DAG, W_DAG, M, Z))
    print('KL5 ', KL5(data, B_DAG, W_DAG, M, Z))
    

    print('KL2 ', KL2(train_data, test_data, B_DAG, W_DAG, M, Z))
    print('KL3 ', KL3(train_data, test_data, B_DAG, W_DAG, M, Z))
    print('KL4 ', KL4(train_data, test_data, B_DAG, W_DAG, M, Z))
    print('KL5 ', KL5(train_data, test_data, B_DAG, W_DAG, M, Z))

    print('=========================')
    print('Ground truth \n', W_DAG)

    
    print('Least Square \n', numbers_to_KL_divergence(1))
    print('Batch_least_square_median \n', numbers_to_KL_divergence(2))
    print('Batch_least_square_mean \n',numbers_to_KL_divergence(3))
    print('Cauchy Tree \n',numbers_to_KL_divergence(4))
    print('Cauchy General \n',numbers_to_KL_divergence(5))
    '''



