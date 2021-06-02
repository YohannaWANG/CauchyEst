#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 13:46:02 2021

@author: yohanna
"""

'''
data arguments
n: number of nodes
s: number of samples
d: average degree of node
'''
n = 100
s = 2000
d = 5
ill = 2

batch = 30

'''
data arguments
choice: choice which real bnlearn data to load. 
load: choice whether load synthetic data or real bnlearn data
options: 'syn', 'real'
options: 'ecoli70', 'magic-niab', 'magic-irri', 'arth150', #{'healthcare', 'sangiovese', 'mehra'}
'''
load = 'syn'
choice = 'ecoli70'
train = 0.9
test = 0.1

'''
data arguments
tg: type of graph, options: chain, er, sf, rt
tn: type of noise, options: ev, uv, ca, ill, exp, gum
th: threshold for weighted matrix
'''
tg = 'er' 
tn = 'uv'
th = 0.3

'''
data arguments
sf: scaling factor for range of binary DAG adjacency
c: number of chain components
dt: determinant of covariance matrix
'''
sf = 1.0
c = 3
dt = 1.0


'''
loss arguments=str, default=algorithm, help='choose structure learning algorithm')

ev: whether to assume equal variances for likelihood objective
l1: coefficient of L1 penalty of DAG matrix
l2: coefficient of structure penalty
k1: coefficient of L1 penalty of undirected matrix
'''
ev = True
l1 = 2e-2
l2 = 1e-4
k1 = 1e-3
eta = 1


'''
train arguments
lr: learning rate of Adam optimiser
ep: number of epochs of training
ck: number of iterations between each checkpoint
'''
lr = 1e-3
ep = 5000
ck = 5000


'''
miscellaneous arguments
seed: seed value for randomness
log: log on file (True) or print on console (False)
gpu: gpu number, options: 0, 1, 2, 3, 4, 5, 6, 7
'''
seed = 5
log = False
gpu = 0


import argparse
def parse():
    '''add and parse arguments / hyperparameters'''
    p = argparse.ArgumentParser()
    p = argparse.ArgumentParser(description="Chain Graph Structure Learning from Observational Data")
    
    p.add_argument('--n', type=int, default=n, help='number of nodes')
    p.add_argument('--s', type=int, default=s, help='number of samples')
    p.add_argument('--d', type=int, default=d, help='average degree of node')
    p.add_argument('--batch', type=int, default=batch, help='number of batch size')
    p.add_argument('--train', type=float, default=train, help='the proportion of training data')
    p.add_argument('--test', type=float, default=test, help='the proportion of testing data')
    p.add_argument('--ill', type=int, default=ill, help='number of ill conditioned nodes')
    
    p.add_argument('--choice', type=str, default=choice, help='choose which real bnlearn data to load')
    p.add_argument('--load', type=str, default=load, help='either load synthetic or real data')


    p.add_argument('--tg', type=str, default=tg, help='type of graph, options: er, sf')
    p.add_argument('--tn', type=str, default=tn, help='type of noise, options: ev, uv, exp, gum')
    p.add_argument('--th', type=float, default=th, help='threshold for weighted matrix')

    p.add_argument('--sf', type=float, default=sf, help='scaling factor for range of binary DAG adjacency')    
    p.add_argument('--c', type=int, default=c, help='number of chain components')
    p.add_argument('--dt', type=float, default=dt, help='determinant of covariance matrix')

    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ('no', 'false', 'f', 'n', '0'): return False
        else: return True

    p.add_argument("--ev", type=str2bool, default=ev, help="whether to assume equal variances for likelihood objective")
    p.add_argument('--l1', type=float, default=l1, help='coefficient of L1 penalty of DAG matrix')
    p.add_argument('--l2', type=float, default=l2, help='coefficient of structure penalty')
    p.add_argument('--k1', type=float, default=k1, help='coefficient of L1 penalty of undirected matrix')
    p.add_argument('--eta', type=float, default=eta, help='coefficient of structure penalty (h)')

    p.add_argument('--lr', type=float, default=lr, help='learning rate of Adam optimiser')
    p.add_argument('--ep', type=int, default=int(ep), help='number of epochs of training')
    p.add_argument('--ck', type=int, default=ck, help='number of iterations between each checkpoint')
    
    p.add_argument('--seed', type=int, default=seed, help='seed value for randomness')
    p.add_argument("--log", type=str2bool, default=log, help="log on file (True) or print on console (False)")
    p.add_argument('--gpu', type=int, default=gpu, help='gpu number, options: 0, 1, 2, 3, 4, 5, 6, 7')
    
    p.add_argument('-f') # for jupyter default
    return p.parse_args()



import os, inspect, logging, uuid
class Logger():
    def __init__(self, p):
        '''Initialise logger '''

        # setup log checkpoint directory
        current = os.path.abspath(inspect.getfile(inspect.currentframe()))
        Dir = os.path.join(os.path.split(os.path.split(current)[0])[0], "checkpoints")
        self.log = p.log
        
        # setup log file  
        if self.log:       
            if not os.path.exists(Dir): os.makedirs(Dir)
            name = str(uuid.uuid4())
            
            Dir = os.path.join(Dir, name)
            if not os.path.exists(Dir): os.makedirs(Dir)
            p.dir = Dir

            # setup logging
            logger = logging.getLogger(__name__)
            
            file = os.path.join(Dir, name + ".log")
            logging.basicConfig(format="%(asctime)s - %(levelname)s -   %(message)s", filename=file, level=logging.INFO)
            self.logger = logger
        

    # function to log
    def info(self, s):
        if self.log: self.logger.info(s)
        else: print(s)



import torch, numpy as np, random
def setup():
    
    # parse arguments
    p = parse()
    p.logger = Logger(p)
    D = vars(p)


    # log configuration arguments
    l = ['']*(len(D)-1) + ['\n\n']
    p.logger.info("Arguments are as follows.")
    for i, k in enumerate(D): p.logger.info(k + " = " + str(D[k]) + l[i]) 


    # set seed
    s = p.seed
    random.seed(s)

    torch.manual_seed(s)
    p.gen = torch.Generator().manual_seed(s)

    np.random.seed(s)
    p.rs = np.random.RandomState(s)
    os.environ['PYTHONHASHSEED'] = str(s) 


    # set device (gpu/cpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
    os.environ["CUDA_VISIBLE_DEVICES"] = str(p.gpu)
    p.device = torch.device('cuda') if p.gpu != '-1' and torch.cuda.is_available() else torch.device('cpu')   

    return p