#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:43:36 2021

@author: yohanna
"""
import numpy as np
import matplotlib.pyplot as plt

from data import SynDAG
from evl import DCP, DKL_ud

from methods import clime, glasso_R, empirical_est, \
                    least_square, batch_least_square_mean, batch_least_square_median,  \
                    CauchyEst_Tree, CauchyEst_General
import config
p = config.setup()
lgr = p.logger


sample = np.array([1000, 2000, 3000, 4000, 5000])

def main():
    
    KL_GLASSO = []
    KL_EMP    = []
    KL_CLIME  = [] 
    
    KL_GLASSO_STD = []
    KL_EMP_STD    = []
    KL_CLIME_STD  = [] 

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
        kl_glasso = []
        kl_emp = []
        kl_clime = []
        
        kl_ls = []
        kl_batch_med = []
        kl_batch_mean = []
        kl_cau_tree = []
        kl_cau_gen = []
        
        for j in range(1):

            p.s = sample[i]
            Input = SynDAG(p)
            Input.visualise()
            plt.show()
            
            W_DAG = Input.A
            B_DAG = Input.B
            data = Input.X
            
            Z = Input.Z
            I = np.identity(p.n)
            A = np.linalg.inv(np.transpose(I - W_DAG))
            B = np.linalg.inv(I - W_DAG)
            M = np.matmul(np.matmul(A, Z), B)

            
            print('GLASSO')
            cov_glasso_est = glasso_R(data)
            kl_g = DKL_ud(cov_glasso_est, M)   
            kl_glasso.append(kl_g)
            
            print('EMP')
            cov_emp_est = empirical_est(data)
            kl_e = DKL_ud(cov_emp_est, M)  
            kl_emp.append(kl_e)
            
            print('CLIME')
            cov_clime_est  = clime(data)
            kl_c = DKL_ud(cov_clime_est, M)
            kl_clime.append(kl_c)
            

            print('LEAST SQUARE')
            A_est_ls = least_square(data, B_DAG)
            kl_l = DCP(data, W_DAG, A_est_ls, M, Z)
            kl_ls.append(kl_l)


            print('LS Median Batch 20')
            A_est_ls_batch_med_20 = batch_least_square_median(data, B_DAG, 20)
            kl2 = DCP(data, W_DAG, A_est_ls_batch_med_20, M, Z)
            kl_batch_med.append(kl2)

            print('LS Mean Batch 20')
            A_est_ls_batch_mean_20 = batch_least_square_mean(data, B_DAG, 20)
            kl3 = DCP(data, W_DAG, A_est_ls_batch_mean_20, M, Z)
            kl_batch_mean.append(kl3)


            print('Cauchy Tree')
            A_est_cau_med = CauchyEst_Tree(data, B_DAG)
            kl_ct = DCP(data, W_DAG, A_est_cau_med, M, Z)
            kl_cau_tree.append(kl_ct)

            print('Cauchy General')
            A_est_he_med = CauchyEst_General(data, B_DAG)
            kl_cg = DCP(data, W_DAG, A_est_he_med, M, Z)
            kl_cau_gen.append(kl_cg)


                
        ''' GLASSO  '''
        kl_glasso_est = np.median(kl_glasso)
        kl_glasso_std = np.std(kl_glasso)
        
        KL_GLASSO.append(kl_glasso_est)
        KL_GLASSO_STD.append(kl_glasso_std)
        
        
        ''' Empirical estimator'''
        kl_emp_est = np.median(kl_emp)
        kl_emp_std = np.std(kl_emp)
        
        KL_EMP.append(kl_emp_est)
        KL_EMP_STD.append(kl_emp_std)
        
        
        ''' CLIME'''
        kl_clime_est = np.median(kl_clime)
        kl_clime_std = np.median(kl_clime)
        
        KL_CLIME.append(kl_clime_est)
        KL_CLIME_STD.append(kl_clime_std)

        ''' Least square'''        
        kl_ls_est = np.median(kl_ls)
        kl_ls_std = np.std(kl_ls)

        KL_LS.append(kl_ls_est)
        KL_LS_STD.append(kl_ls_std)
        
        
        ''' Least square batch median'''
        kl_batch_med_est = np.mean(kl_batch_med)
        kl_batch_med_std = np.std(kl_batch_med)

        KL_BATCH_MED.append(kl_batch_med_est)
        KL_BATCH_MED_STD.append(kl_batch_med_std)
        

        ''' Least square batch mean'''
        kl_batch_mean_est = np.mean(kl_batch_mean)
        kl_batch_mean_std = np.std(kl_batch_mean)

        KL_BATCH_MEAN.append(kl_batch_mean_est)
        KL_BATCH_MEAN_STD.append(kl_batch_mean_std)
        

        '''Cauchy Tree '''
        kl_cau_tree_est = np.mean(kl_cau_tree)
        kl_cau_tree_std = np.std(kl_cau_tree)

        KL_CAU_TREE.append(kl_cau_tree_est)
        KL_CAU_TREE_STD.append(kl_cau_tree_std)
        

        '''Cauchy General '''
        kl_cau_gen_est = np.mean(kl_cau_gen)
        kl_cau_gen_std = np.std(kl_cau_gen)

        KL_CAU_GEN.append(kl_cau_gen_est)
        KL_CAU_GEN_STD.append(kl_cau_gen_std)
        

    print('KL_GLASSO = ', KL_GLASSO)
    print('KL_EMP = ', KL_EMP)
    print('KL_CLIME = ', KL_CLIME)

    print('KL_LS = ', KL_LS)
    print('KL_BATCH_MED = ', KL_BATCH_MED)
    print('KL_BATCH_MEAN = ', KL_BATCH_MEAN)
    print('KL_CAU_TREE = ', KL_CAU_TREE)
    print('KL_CAU_GEN = ', KL_CAU_GEN)
    

    print('KL_GLASSO_STD = ', KL_GLASSO_STD)
    print('KL_EMP_STD = ', KL_EMP_STD)
    print('KL_CLIME_STD = ', KL_CLIME)

    print('KL_LS_STD = ', KL_LS)
    print('KL_BATCH_MED_STD = ', KL_BATCH_MED)
    print('KL_BATCH_MEAN_STD = ', KL_BATCH_MEAN)
    print('KL_CAU_TREE_STD = ', KL_CAU_TREE)
    print('KL_CAU_GEN_STD = ', KL_CAU_GEN)

    plt.figure(figsize=(18, 10))
    
    plt.errorbar(sample, KL_GLASSO, yerr=KL_GLASSO_STD, label='GLASSO', linewidth=4)
    plt.errorbar(sample, KL_EMP, yerr=KL_EMP_STD, label='MLE', linewidth=4)
    plt.errorbar(sample, KL_CLIME, yerr=KL_CLIME_STD, linestyle='--', dashes=(5, 8), label='CLIME', linewidth=4)
    plt.errorbar(sample, KL_LS, yerr=KL_LS_STD, label='LS', linewidth=3)
    plt.errorbar(sample, KL_BATCH_MED, yerr=KL_BATCH_MED_STD, label='BatchMed_LS_20', linewidth=4)
    plt.errorbar(sample, KL_BATCH_MEAN, yerr=KL_BATCH_MEAN_STD, linestyle='--', dashes=(5, 5),label='BatchAvg_LS+20', linewidth=4)
    plt.errorbar(sample, KL_CAU_TREE, yerr=KL_CAU_TREE_STD, label='CauchyEstTree', linewidth=4)
    plt.errorbar(sample, KL_CAU_GEN, yerr=KL_CAU_GEN_STD, label='CauchyEst', linewidth=4)

    plt.legend(loc='upper right', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.xlabel('Sample size (100 nodes, Random Tree) ', fontsize=30)
    plt.ylabel('KL_Divergence', fontsize=35)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()

