## ----chapter2-init-stanza, include = FALSE, cache = FALSE, echo = FALSE-------
library(rbmn)
library(Rgraphviz)
library(igraph)
library(lattice)
library(bnlearn)


## ----inc-da--bnlearn, echo = TRUE---------------------------------------------
library(bnlearn)
dag.bnlearn <- model2network("[G][E][V|G:E][N|V][W|V][C|N:W]")
dag.bnlearn


## ----marginal-independence----------------------------------------------------
crop.nodes <- nodes(dag.bnlearn)
for (n1 in crop.nodes) {
  for (n2 in crop.nodes) {
    if (dsep(dag.bnlearn, n1, n2))
      cat(n1, "and", n2, "are independent.\n")
  }#FOR
}#FOR


## ----dsep-single--------------------------------------------------------------
dsep(dag.bnlearn, "V", "V")


## ----conditional-indep--------------------------------------------------------
for (n1 in crop.nodes[crop.nodes != "V"]) {
  for (n2 in crop.nodes[crop.nodes != "V"]) {
    if (n1 < n2) {
      if (dsep(dag.bnlearn, n1, n2, "V"))
        cat(n1, "and", n2, "are independent given V.\n")
    }#THEN
  }#FOR
}#FOR


## ----self-independence--------------------------------------------------------
dsep(dag.bnlearn, "E", "V", "V")


## ----path---------------------------------------------------------------------
path.exists(dag.bnlearn, from = "E", to = "C")


## ----probabilities-and-bundle-------------------------------------------------
E.dist <- list(coef = c("(Intercept)" = 50), sd = 10)
G.dist <- list(coef = c("(Intercept)" = 50), sd = 10)
V.dist <- list(coef = c("(Intercept)" = -10.35534,
                        E = 0.70711, G = 0.5), sd = 5)
N.dist <- list(coef = c("(Intercept)" = 45, V = 0.1), sd = 9.949874)
W.dist <- list(coef = c("(Intercept)" = 15, V = 0.7), sd = 7.141428)
C.dist <- list(coef = c("(Intercept)" = 0, N = 0.3, W = 0.7), sd = 6.25)
dist.list = list(E = E.dist, G = G.dist, V = V.dist,
                 N = N.dist, W = W.dist, C = C.dist)


## ----assemble-fitted-network--------------------------------------------------
gbn.bnlearn <- custom.fit(dag.bnlearn, dist = dist.list)


## ----data-for-the-chapter, echo = FALSE---------------------------------------
set.seed(4567)
cropdata200 <- rbn(gbn.bnlearn, n = 200)
set.seed(1234)
cropdata20k <- rbn(gbn.bnlearn, n = 20000)


## ----print-node-G-and-C-------------------------------------------------------
gbn.bnlearn$G
gbn.bnlearn$C


## ----load-rbmn-and-import-----------------------------------------------------
library(rbmn)
gbn.rbmn <- bnfit2nbn(gbn.bnlearn)


## ----rbmn-compute-params------------------------------------------------------
gema.rbmn <- nbn2gema(gbn.rbmn)
mn.rbmn <- gema2mn(gema.rbmn)
print8mn(mn.rbmn)


## ----rbmn-structure-----------------------------------------------------------
str(mn.rbmn)


## ----showcase-data------------------------------------------------------------
dim(cropdata200)
round(head(cropdata200), 2)


## ----learn-gbn-from-data------------------------------------------------------
crop.fitted <- bn.fit(dag.bnlearn, data = cropdata200)


## ----replace-parameters-C, eval = FALSE---------------------------------------
## crop.fitted$C <- lm(C ~ N + W, data = cropdata200)


## ----show-penalised, eval = FALSE---------------------------------------------
## library(penalized)
## crop.fitted$C <- penalized(C ~ N + W, lambda1 = 0, lambda2 = 1.5,
##                    data = cropdata200)


## ----show-parameters-E--------------------------------------------------------
crop.fitted$E


## ----show-parameters-C--------------------------------------------------------
crop.fitted$C


## ----replace-paramters-C-for-real---------------------------------------------
crop.fitted$C <- lm(C ~ N + W - 1, data = cropdata200)
crop.fitted$C


## ----compare-with-lm----------------------------------------------------------
lmC <- lm(C ~ N + W, data = cropdata200[, c("N", "W", "C")])
coef(lmC)


## ----confidence-intervals-----------------------------------------------------
confint(lmC)


## ----correlation-matrix-------------------------------------------------------
cormat <- cor(cropdata200[, c("C", "W", "N")])


## ----precision-matrix---------------------------------------------------------
library(corpcor)
invcor <- cor2pcor(cormat)
dimnames(invcor) <- dimnames(cormat)
invcor


## ----conditional-independence-cor---------------------------------------------
ci.test("C", "W", "N", test = "cor", data = cropdata200)


## ----structure-learning-from-data---------------------------------------------
pdag1 <- iamb(cropdata200, test = "cor")


## ----learned-with-iamb, echo = FALSE, results = "hide"------------------------
local({
  pdf("figures/learning2.pdf", height = 4.5, width = 6, paper = "special");
  gR <- graphviz.plot(pdag1, render = FALSE)
  nodeRenderInfo(gR)$fontsize = 11
  renderGraph(gR)
  dev.off()
})


## ----learning-with-whitelist--------------------------------------------------
wl <- matrix(c("V", "N"), ncol = 2)
wl
pdag2 <- iamb(cropdata200, test = "cor", whitelist = wl)
all.equal(dag.bnlearn, pdag2)


## ----learning-with-more-data--------------------------------------------------
dim(cropdata20k)
pdag3 <- iamb(cropdata20k, test = "cor")
all.equal(dag.bnlearn, pdag3)


## ----compare-scores-----------------------------------------------------------
score(dag.bnlearn, data = cropdata20k, type = "bic-g")
score(dag.bnlearn, data = cropdata20k, type = "bge")


## ----print-rbmn---------------------------------------------------------------
print8nbn(gbn.rbmn)


## ----print-gema---------------------------------------------------------------
print8gema(gema.rbmn)


## ----print-conditional-joints-------------------------------------------------
print8mn(condi4joint(mn.rbmn, par = "C", pour = "V", x2 = 80))
print8mn(condi4joint(mn.rbmn, par = "V", pour = "C", x2 = 80))


## ----print-conditional-joints-v2----------------------------------------------
unlist(condi4joint(mn.rbmn, par = "C", pour = "V", x2 = NULL))


## ----hidden-random-seed, echo = FALSE-----------------------------------------
set.seed(2345)

## ----random-simulation--------------------------------------------------------
nobs <- 4
VG <- rnorm(nobs, mean = 50, sd = 10)
VE <- rnorm(nobs, mean = 50, sd = 10)
VV <- rnorm(nobs, mean = -10.355 + 0.5 * VG + 0.707 * VE, sd = 5)
VN <- rnorm(nobs, mean = 45 + 0.1 * VV, sd = 9.95)
cbind(VV, VN)


## ----hidden-seed-again, echo = FALSE------------------------------------------
set.seed(1234)

## ----simulation-with-rbn------------------------------------------------------
sim <- rbn(gbn.bnlearn, n = nobs)
sim[, c("V", "N")]


## ----data2, results = "hide", include = TRUE----------------------------------
set.seed(4567)
cropdata200 <- rbn(gbn.bnlearn, n = 200)
set.seed(1234)
cropdata20k <- rbn(gbn.bnlearn, n = 20000)


## ----simulation-with-cpdist---------------------------------------------------
head(cpdist(gbn.bnlearn, nodes = c("C", "N", "W"), evidence = (C > 80)))


## ----simulation-with-cpdist-v2------------------------------------------------
head(cpdist(gbn.bnlearn, nodes = "V",
            evidence = list(G = 10, E = 90), method = "lw"), n = 5)


## ----simulation-with-cpquery--------------------------------------------------
cpquery(gbn.bnlearn, event = (V > 70),
        evidence = list(G = 10, E = 90), method = "lw")


## ----load-and-set-igraph------------------------------------------------------
library(igraph)
igraph.options(print.full = TRUE)
dag0.igraph <- graph.formula(G-+V, E-+V, V-+N, V-+W, N-+C, W-+C)
dag0.igraph


## ----graph-to-igraph----------------------------------------------------------
dag.igraph <- as.igraph(dag.bnlearn)


## ----igraph-nodes-and-arcs----------------------------------------------------
V(dag.igraph)
E(dag.igraph)


## ----igraph-for-the-figure, fig.keep = "none"---------------------------------
par(mfrow = c(1, 3), mar = rep(3, 4), cex.main = 2)
plot(dag.igraph, main = "\n1: defaults")
ly <- matrix(c(2, 3, 1, 1, 2, 3, 1, 4, 4, 2, 3, 2), 6)
plot(dag.igraph, layout = ly, main = "\n2: positioning")
vcol <- c("black", "darkgrey", "darkgrey", rep(NA, 3))
lcol <- c(rep("white", 3), rep(NA, 3))
par(mar = rep(0, 4), lwd = 1.5)
plot(dag.igraph, layout = ly, frame = TRUE, main = "\n3: final",
     vertex.color = vcol, vertex.label.color = lcol,
     vertex.label.cex = 3, vertex.size = 50,
     edge.arrow.size = 0.8, edge.color = "black")


## ----actual-1x3-figure, echo = FALSE, results = "hide"------------------------
local({
  pdf("figures/igraph4.pdf", width = 12, height = 5)
  par(mfrow = c(1, 3), mar = rep(3, 4), cex.main = 2)
  plot(dag.igraph, main = "\n1: defaults",
       vertex.size = 40, vertex.label.cex = 2, vertex.color = "grey",
       vertex.label.color = "black")
  ly <- matrix(c(2, 3, 1, 1, 2, 3, 1, 4, 4, 2, 3, 2), 6)
  plot(dag.igraph, layout = ly, main = "\n2: positioning",
       vertex.size = 40, vertex.label.cex = 2, vertex.color = "grey",
       vertex.label.color = "black")
  vcol <- c("black", "darkgrey", "darkgrey", rep(NA, 3))
  lcol <- c(rep("white", 3), rep(NA, 3))
  par(mar = rep(0, 4), lwd = 1.5)
  plot(dag.igraph, layout = ly, frame = TRUE,
       main = "\n3: final",
       vertex.color = vcol, vertex.label.color = lcol,
       vertex.label.cex = 3, vertex.size = 50,
       edge.arrow.size = 0.8, edge.color = "black")
  dev.off()
})


## ----bnlearn-quantile-plot, fig.keep = "none"---------------------------------
gbn.fit <- bn.fit(dag.bnlearn, cropdata20k)
bn.fit.qqplot(gbn.fit)


## ----bnlearn-quantile-plot-node, fig.keep = "none"----------------------------
bn.fit.qqplot(gbn.fit$V)


## ----lattice-plot-no-residuals-error, fig.keep = "none"-----------------------
bn.fit.qqplot(gbn.bnlearn)


## ----hidden-function-declaration, echo = FALSE--------------------------------
condi4joint <- function(...) {
  res <- rbmn::condi4joint(...)
  res$rho = zapsmall(res$rho)
  return(res)
}


## ----rbmn-closed-form-distribution--------------------------------------------
C.EV <- condi4joint(mn.rbmn, par = "C", pour = c("E", "V"), x2 = NULL)
C.EV$rho


## ----check-corresponding-dseparation------------------------------------------
dsep(gbn.bnlearn, "E", "C", "V")


## ----three-dimensional-plot, eval = FALSE-------------------------------------
## set.seed(5678)
## cropdata3 <- cpdist(gbn.bnlearn, nodes = c("E", "V", "C"),
##                     evidence = TRUE, n = 1000)
## plot(cropdata3$V, cropdata3$C, type = "n",
##      main = "C | V, E; E is the point size")
## cexlim <- c(0.1, 2.4)
## cexE <- cexlim[1] + diff(cexlim) / diff(range(cropdata3$E)) *
##                     (cropdata3$E - min(cropdata3$E))
## points(cropdata3$V, cropdata3$C, cex = cexE)
## cqa <- quantile(cropdata3$C, seq(0, 1, 0.1))
## abline(h = cqa, lty = 3)


## ----cpdist-simulation-plot, echo = FALSE, results = "hide"-------------------
local({
  pdf("figures/cpdist1.pdf")
  set.seed(5678)
  cropdata3 <- cpdist(gbn.bnlearn, nodes = c("E", "V", "C"),
                      evidence = TRUE, n = 1000)
  plot(cropdata3$V, cropdata3$C, type="n",
       main = "C | V , E; E is the point size")
  cexlim <- c(0.1, 2.4)
  cexE <- cexlim[1] + diff(cexlim) / diff(range(cropdata3$E)) *
    (cropdata3$E - min(cropdata3$E))
  points(cropdata3$V, cropdata3$C, cex = cexE)
  cqa <- quantile(cropdata3$C, seq(0, 1, 0.1))
  abline(h = cqa, lty = 3)
  dev.off()
  pdf("figures/cpdist2.pdf")
  plot(cropdata3$E, cropdata3$C, type="n",
       main = "C | V , E; V is the point size")
  cexV <- cexlim[1] + diff(cexlim) / diff(range(cropdata3$V)) *
    (cropdata3$V - min(cropdata3$V))
  points(cropdata3$E, cropdata3$C, cex=cexV)
  abline(h = cqa, lty = 3)
  dev.off()
})

