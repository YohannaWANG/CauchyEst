
<img align="left" src="docs/images/Cauchy_est_logo.png"> &nbsp; &nbsp;

   

# Learning Sparse Fixed-Structure Gaussian Bayesian Networks



## Introduction
Gaussian Bayesian networks (a.k.a.\ linear Gaussian structural equation models) are widely used to model causal interactions among continuous variables.
In this work, we study the problem of **learning a fixed-structure Gaussian Bayesian network up to a bounded error in total variation distance.** 

We analyze the commonly used node-wise least squares regression **{LeastSquares}** and prove that it has the near-optimal sample complexity.
We also propose a new estimator **{CauchyEst}** based on some interesting properties of Cauchy random variables, and prove near-optimal sample complexity for polytrees.
Experimentally, we show that **{CauchyEst}** and its extension **{CauchyEstGeneral}** compare favorably to **{LeastSquares}**.                               

## Example
<img width="820" align="center" src="docs/images/example.png"> 



## Prerequisites

- **Python 3.6+**
   - `networkx`
   - `argpase`
   - `itertools`
   - `numpy`
   - `scipy`
   - `sklearn`
   - `matplotlib`
   - `torch`: Optional.
- **R 4.0.0**
   - `rpy2`: Python interface, enables calling R from Python. Install [rpy2](https://pypi.org/project/rpy2/) first.
   - `bnlearn` : [Bayesian network learning and inference](bnlearn.com) 
   - `glasso` : [Graphical Lasso: Estimation of Gaussian Graphical Models](https://cran.r-project.org/web/packages/glasso/index.html)
   - `flare`: [Family of Lasso Regression](https://cran.r-project.org/web/packages/flare/index.html)

## Contents

- **Data**  - Real Bayesian network data from bnlearn;
- `data.py` - synthetic chain graph data, including graph simulation and data simulation. Load real Bnlearn data 
- `evaluate.py` - algorithm accuracy evaluation 
- `config.py` - Set parameters for Bayesian network (eg. node number, graph degree, sample size)
- `utils.py` - simulation parameters, such as selecte graph type, node number, data type, graph degree, etc.  
- `utils.R` - load bnlearn graph; Run CLIME algorithm


## Running a simple demo

The simplest way to try out DCOV is to run a simple example:
```bash
$ git clone https://github.com/YohannaWANG/CauchyEst.git
$ cd CauchyEst/
$ python CauchyEst/demo.py
```

## Runing as a command

Alternatively, if you have a CSV data file `X.csv`, you can install the package and run the algorithm as a command:
```bash
$ pip install git+git://github.com/YohannaWANG/CauchyEst
$ cd CauchyEst
$ python main.py --regress_method mgcv --n 50 --s 1000 --d 4 
```

## Algorithms
- ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+)  **Algorithm 1** states our two-phased recovery approach. We estimate the coefficients of the Bayesian network in the first phase and use them to recover the variances in the second phase.
   <img width="800" align ="center" alt="characterization" src="/docs/images/algo1.png" >
- ![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) **Algorithm 2** is recovering the coefficients in a Bayesian network using a linear least squares estimator. 
   <img width="800" align ="center" alt="characterization" src="/docs/images/algo2.png">    
- ![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) **Algorithm 3** is our CauchyEst algorithm for variable with p parents.
   <img width="800" align ="center" alt="characterization" src="/docs/images/algo3.png" >  
- ![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) **Algorithm 4** is our CauchyEst algorithm for recovering the coefficients in polytree Bayesian networks.
   <img width="800" align ="center" alt="characterization" src="/docs/images/algo4.png" >    
- ![#d03c15](https://via.placeholder.com/15/d03c15/000000?text=+) **Algorithm 5** extend CauchyEst algorithm to general Bayesian networks.
   <img width="800" align ="center" alt="characterization" src="/docs/images/algo5.png" >    

## Performance

100 nodes, degree 5, ER graph     | Effect of changing batch size, 100 nodes, Random Tree graph
:--------------------------------------------------------------------:|:-----------------------------------------------------------------------------------:
<img width="400" alt="characterization" src="/docs/images/100_node_syn_uv_d5_ER.png" >  |  <img width="400" alt="characterization" src="/docs/images/Batch_LS_100node_randomTree.png" >



## Open questions
-The most prominent question raised by our work is whether we can rigorously explain the superior performance of **{CauchyEst}* and **{CauchyEstGeneral}** in the experiments. 
- One can view **{LeastSquares}** as a special case of **{CauchyEst}** by using only a **{single}** batch (with >>p samples) and enforcing Algorithm 3 to use the least squares solution. It would be interesting to see if one can design an algorithm that interpolates between **{LeastSquares}** and **{CauchyEst}** while exhibiting a tradeoff between `number of batches` and `sample complexity`. Some preliminary experiments on this trade-off are provided in the supplementary material.
- The current work only analyzes the sample complexity in the case that the distributions are realizable by the given structure. It remains an open question to guarantee bounds on the error in the non-realizable setting, i.e., to find the distribution that is best fitted by the given structure. 

## Citation
If you use any part of this code in your research or any engineering project, please cite our paper:

## Contacts

Please feel free to contact us if you meet any problem when using this code. We are glad to hear other advise and update our work. 
We are also open to collaboration if you think that you are working on a problem that we might be interested in it.
Please do not hestitate to contact us!







## TODO tasks (eval.py, algo.py, and main.py)

```diff
+ (Done) General synthetic SEM data;
+ (Done) (FIXED) <**R language**>: bnlearn R data
+ (Done) Tree structure synthetic data;
 
+ (Done) DAG: Liear regression algorithm;
+ (Done) DAG: Least square algorithm;
+ (Done) Undirected graph: GLASSO algotirhm;
+ (Done) Undirected graph: empirical estimator;
 
+ (Done) Performance evaluation (KL-distance) on DAG;
+ (Done) Performance evaluation (KL-distance) on Undirected graph;

+ (Done) Overleaf: algorithm 1;
+ (Done) Overleaf: algorithm 2;
+ (Done) Overleaf: algorithm 3;
+ (Done) Overleaf: algorithm 4;
+ (Done) Add R CLIME & TIGER algorithm;
+ (Done) Code for generate plot;
+ (Done) Data: synthetic ill-conditioned data;
+ (Done) Experiments: Ill-conditioned models. Like say one of the variables has noise variance very close to 0;
+ (Done) Experiments: how the error decreases for empirical and GLASSO on separate plots;


- (Done) For distributions generated by degree 10 Bayes network, run our three algorithms with d=5. See how fast the errors converge.
- (Done) Real datasets from bnlearn
- (Done) Ill-conditioned models. Like say one of the variables has noise variance very close to 0. Hopefully, here we can also find some difference between CauchyEst and CauchyEstGeneral;
- (Done) some fraction have  N = r.normal(scale = 10^-10) others have N = r.normal(scale = 1.0) as before;
- (Done) for each server job, try running on "1 algorithm, 1 parameter setting, X samples". that way you can parallelize quite a lot and you can plot each graph line (with error bars) as soon as one of these jobs complete.
```




