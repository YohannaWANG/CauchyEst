" 
Function: Load real Gaussian Bayesian Networks from bnlearn
   Dataset is available at: https://www.bnlearn.com/bnrepository/
  
Including: 
1. Gaussian Bayesian networks
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

2. Conditional Linear Gaussian Bayesian Networks
  Small networks(<20 nodes):
  > HEALTHCARE (7 nodes, 9 arcs, 42 parameters)
  > SANGIOVESE (15 nodes, 55 arcs, 259 parameters)
  Details: https://www.bnlearn.com/bnrepository/clgaussian-small.html#healthcare
  
  Medium networks(20-50 nodes)
  > MEHRA (24 nodes, 71 arcs, 324423 parameters)
  Details: https://www.bnlearn.com/bnrepository/clgaussian-medium.html#mehra

NOTES: I didn't add other 22 discrete bayesian network datasets

Author: Yohanna
Date: 2021-05-23
Email: yohanna.wang0924@gmail.com
"

load_data <- function(choice){
  library(bnlearn)
  
  ### Gaussian Bayesian networks
  
  # Load medium network
  if (choice == "ecoli70"){
    # This will load the bn object
    # load("ecoli70.rda") 
    network <- readRDS("Data/ecoli70.rds")
  }
  else if(choice == "magic-niab"){
    network <- readRDS("Data/magic-niab.rds")
  }
  # Load large network
  else if(choice == "magic-irri"){
    network <- readRDS("Data/magic-irri.rds")
  }
  # Load very large network
  else if(choice == "arth150"){
    network <- readRDS("Data/arth150.rds")
  }
  
  ###  Conditional Linear Gaussian Bayesian Networks
  
  # Load small network
  else if(choice == "healthcare"){
    network <- readRDS("Data/healthcare.rds")
  }
  
  else if(choice == "sangiovese"){
    network <- readRDS("Data/sangiovese.rds")
  }
  
  # Load medium networks (with no NAN parameters)
  else if(choice == "mehra"){
    network <- readRDS("Data/sangiovese.rds")
  }
  return(network)
}

get_graph <- function(choice){ 
  library(bnlearn)
  network <- load_data(choice)
  
  " Get binary adj matrix and extract the coefficients between all parents and children"
  G_binary <- amat(network)
  coef <- coefficients(network)
  nodes <- unlist(dimnames(G_binary)[1])
  
  n <- nrow(G_binary)
  p <- ncol(G_binary)

  " X = AX + b 
  Initialize an all-zero matrix to store inception value (b) "
  interception <- matrix(0, n, p)
  colnames(interception) <- nodes
  rownames(interception) <- nodes
  
  " Initialize an zero matrix for weighted adjacency matrix (A)" 
  G_weighted <- matrix(0, n, p)
  colnames(G_weighted) <- nodes
  rownames(G_weighted) <- nodes
  
  for (i in 1:nrow(G_binary)) {
    " Then find each coefficient pairs" 
    temp_i <- coef[i]

    " Name of the children " 
    col_name <- names(coef[i])
    row_name <- names(coef[i])

    " Given children node, find the coefficients with its parents"
    rel_i <- unlist(unname(temp_i))
    
    "Given row_name and col_name, now we add the extracted intercept into 
     the initialized interception matrix"
    intercep_temp <- rel_i[1]
    interception[row_name, col_name] <- intercep_temp
    
    if (length(rel_i) <  2){
      " Then we have interception value only"
    }
    if (length(rel_i) >= 2){ 
      
      for (j in 2:length(rel_i)){
        " Get the name of the parent " 
        row_name <- names(rel_i[j])
        a_ij <- unname(rel_i[j])
        
        " Generate weighted adj matrix: add the coefficients into G_weighted " 
        if((!is.na(row_name) ) & (!is.na(col_name))){
          #print(G_binary[row_name, col_name])
          G_weighted[row_name, col_name] <- a_ij
        }
      }
    }
  }
  return(list(G_binary, G_weighted, interception))
}

" Function: Estimate sparse undirected graphical models. i.e. Gaussian precision matrix
      sumg: High-deimensional Sparse Undirected Graphical Models
  Two estimation preceduresbased on column by column regression:
    (1) Tuning-Insensitive Graph Estimation and Regression based on square root Lasso (tiger);
    (2) The Constrained L1 Minimization for Sparse Precision Matrix Estimation using either
        L1 penalty (clime).
  Notes: The optimization algorithm are implemented based on the alternating 
          direction method of multipliers (ADMM) with linearization method and 
          multi-stage screening of variables. 
          Missing values can be tolerated for CLIME in the data matrix.
"
clime <- function(data){
  "Function: run CLIME algorithm"
  
  library('flare')
  out <- sugm(data, method='clime')
  clime_cov <- out$sigma

  return(clime_cov)
}

tiger <- function(data){
  " Function: run TIGER algorithm"
  library('flare')
  out <- sugm(data, method='tiger')
  tiger_cov <- out$sigma

  return(tiger_cov)
}

glasso_r <- function(data){
  library(glasso)
  #data<-matrix(rnorm(3*10),ncol=3)
  s <- var(data)
  a <- glasso(s, rho=0.02)
  glasso_est<-glasso(s,rho=.02, w.init=a$w, wi.init=a$wi)
  glasso_cov <- glasso_est$w
  return(glasso_cov)
}


