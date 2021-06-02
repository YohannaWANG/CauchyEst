#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 10:19:49 2021

@author: yohanna
"""
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import config
p = config.setup()
lgr = p.logger

"""
Function: (Data) Generate synthetic DAG data (linear)
"""

class SynDAG:
    '''class for synthetically-generated directed acyclic graph (DAG) data'''

    def __init__(self, p):
        self.p = p
        if p.load == 'syn':
            B = SynDAG.DAG(p)
            A = SynDAG.WDAG(B, p)

        elif p.load == 'real':
            B, A = SynDAG.load_bnlearn(p)
        else: raise ValueError("No such data load selection")
        
        self.X = SynDAG.SEM(A, p)

        self.A, self.B = A, B

    @staticmethod
    def ER(p):
        '''
        simulate Erdos-Renyi (ER) DAG through networkx package

        Arguments:
            p: argparse arguments

        Uses:
            p.n: number of nodes
            p.d: degree of graph
            p.rs: numpy.random.RandomState

        Returns:
            B: (n, n) numpy.ndarray binary adjacency of ER DAG
        '''
        n, d, s = p.n, p.d, p.rs
        p = float(d)/(n-1)

        G = nx.generators.erdos_renyi_graph(n=n, p=p, seed=None)
        U = nx.to_numpy_matrix(G)
        B = np.tril(U, k=-1)

        return B

    @staticmethod
    def RT(p):
        '''
        simulate Random Tree DAG through networkx package

        Arguments:
        Arguments:
            p: argparse arguments

        Uses:
            p.n: number of nodes
            p.rs: numpy.random.RandomState
        '''
        n, s = p.n, p.s
        G = nx.random_tree(n, seed=s)
        U = nx.to_numpy_matrix(G)
        B = np.tril(U, k=-1)

        return B

    @staticmethod
    def DAG(p):
        '''
        simulate a directed acyclic graph (DAG)

        Arguments:
            p: argparse arguments

        Uses:
            p.n: number of nodes
            p.d: degree of graph
            p.tg: type of graph
            p.rs: numpy.random.RandomState

        Returns:
            B: (n, n) numpy.ndarray binary adjacency of DAG (permuted).
        '''
        t = p.tg.lower()
        if t == 'er':
            B = SynDAG.ER(p)
        elif t == 'rt':
            B = SynDAG.RT(p)
        else:
            raise ValueError(
                "The type of graph is either unknown or has not been implemented")

        # np.random.permutation permutes first axis only
        P = p.rs.permutation(np.eye(p.n))
        B = P.T @ B @ P

        return B

    @staticmethod
    def WDAG(B, p):
        '''
        simulate a weighted directed acyclic graph (DAG)

        Arguments:
            B: binary adjacency of DAG
            p: argparse arguments

        Uses:
            p.sf: scaling factor for range of weights for DAG
            p.rs: numpy.random.RandomState

        Returns:
            A: (n, n) numpy.ndarray weighted adjacency matrix of DAG.
        '''
        A = np.zeros(B.shape)

        s = p.sf
        R = ((-2*s, -0.5*s), (0.5*s, 2*s))
        S = p.rs.randint(len(R), size=A.shape)

        for i, (l, h) in enumerate(R):
            U = p.rs.uniform(low=l, high=h, size=A.shape)
            A += B * (S == i) * U

        return A
    
    @staticmethod
    def load_bnlearn(p):
        """
        Function: Load real bnlearn data
        """
        from rpy2.robjects.packages import importr
        import rpy2.robjects as robjects
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects.vectors import StrVector
        from rpy2.robjects.packages import STAP
    
        import rpy2.robjects.numpy2ri
        from rpy2.robjects import pandas2ri
    
    
        pandas2ri.activate()
        rpy2.robjects.numpy2ri.activate()
        robjects.r.source("utils.R")

    
        with open('utils.R', 'r') as f:
            string = f.read()
    
        bayesian_network = STAP(string, "bayesian_network")
    
    
        """
        Choices including: 
            
        Gaussian Bayesian networks
          Medium networks(20-50 nodes):
          > ECOLI70: (46 nodes, 70 arcs, 162 parameters)
          > MEGIC-NIAB: (44 nodes, 66 arcs, 154 parameters)
          Details: https://www.bnlearn.com/bnrepository/gaussian-medium.html#magic-niab
          
          Large networks(50-100 nodes):
          > MAGIC-IRRI: (64 nodes, 102 arcs, 230 parameters)
          Details: https://www.bnlearn.com/bnrepository/gaussian-large.html#magic-irri
          
          Very large networks(101-1000 nodes):
          > ARTH150: (107 nodes, 150 arcs, 364 parameters)
          Details: https://www.bnlearn.com/bnrepository/gaussian-verylarge.html#arth150 
    
        """
        choice = p.choice
    
        graph_sets = bayesian_network.get_graph(choice)
        
        G_binary = graph_sets[0]
        G_weighted = graph_sets[1]
        interception = graph_sets[2]
        

        data = SynDAG.SEM(G_weighted, p)

        return G_binary, G_weighted 

    @staticmethod
    def SEM(A, p):
        '''
        simulate samples from linear structural equation model (SEM) with specified type of noise.

        Arguments:
            A: (n, n) weighted adjacency matrix of DAG
            p: argparse arguments

        Uses:
            p.n: number of nodes
            p.s: number of samples
            p.rs (numpy.random.RandomState): Random number generator
            p.tn: type of noise, options: ev, nv, exp, gum
                ev: equal variance
                uv: unequal variance
                exp: exponential noise
                gum: gumbel noise

        Returns:
            numpy.ndarray: (s, n) data matrix.
        '''
        s, r, t = p.s, p.rs, p.tn.lower()

        def _SEM(X, I, v, nodes_ill):
            
            '''
            simulate samples from linear SEM for the i-th vertex

            Arguments:
                X (numpy.ndarray): (s, number of parents of vertex i) data matrix
                I (numpy.ndarray): (n, 1) weighted adjacency vector for the i-th node

            Returns:
                numpy.ndarray: (s, 1) data matrix.
            '''
            if t == 'ev':
                N = r.normal(scale=1.0, size=s)
            elif t == 'uv':
                N = r.normal(scale=r.uniform(low=1.0, high=2.0), size=s)
            elif t == 'ca':
                N = np.random.standard_cauchy(size=s)
            elif t == 'ill':
                if v in nodes_ill:
                    N = r.normal(scale=10**-10, size=s)
                else:
                    N = r.normal(scale=1.0, size=s)
            elif t == 'exp':
                N = r.exponential(scale=1.0, size=s)
            elif t == 'gum':
                N = r.gumbel(scale=1.0, size=s)
            else:
                raise ValueError('unknown noise type')

            return X @ I + N
        if p.load == 'syn':
            n = p.n
        elif p.load == 'real':
            n = A.shape[0]
        
        
        X = np.zeros([s, n])
        G = nx.DiGraph(A)
        
        ''' Radomly set ill conditioned nodes'''
        nodes = np.arange(p.n)
        np.random.shuffle(nodes)    
        nodes_ill = nodes[:p.ill]
        
        for v in list(nx.topological_sort(G)):

            P = list(G.predecessors(v))
            X[:, v] = _SEM(X[:, P], A[P, v], v, nodes_ill)

        return X

    def visualise(self, D=[], U=[], name='true'):
        '''
        Arguments:
            D: numpy adjacency of directed edges
            U: list of undirected edges
        '''
        if D == []:
            D = self.A
        n, log = D.shape[1], self.p.log

        def _edges(G, s):
            D = nx.get_edge_attributes(G, s)
            if s != 'weight':
                return [D[k] for k in D.keys()]
            return D

        G = nx.DiGraph(D)
        E = _edges(G, 'weight')
        for (u, v) in E.keys():
            G.add_edge(u, v, width=2, weight=round(E[(u, v)], 2))

        pos = nx.nx_pydot.graphviz_layout(G)
        nx.draw(G, pos=pos, with_labels=True, font_size=10)
        nx.draw_networkx_nodes(G, pos=pos, node_size=500)

        nx.draw_networkx_edges(G, pos=pos, width=_edges(G, 'width'))
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=_edges(G, 'weight'), font_size=6)
        if log:
            plt.savefig(os.path.join(self.p.dir, name+'.png'), dpi=1000)
            


if __name__ == '__main__':
    Input = SynDAG(p)
    Input.visualise()
    W_DAG = Input.A
    B_DAG = Input.B
    data = Input.X 
    print(data)



