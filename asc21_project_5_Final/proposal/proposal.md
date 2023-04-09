# Proposal

## Members

* 김명준
* 장태영
* 현승엽

## Brief Summary of the paper

Introducing a mixed integer optimization (MIO) allows us to solve problems with much larger sizes than what was thought possible in the statistics community (Bertsimas, King and Mazumder, 2016). This paper presented empirical comparisons of three most canonical forms for sparse estimation in a linear model; best subset selection, forward stepwise selection and the lasso. The authors provided an expanded set of simulations to shed more light on the analysis of the previous paper.

## Project Goal

1. Review and implement the details of three methods taking advantage of the materials in the course.
    * Best subset selection : Mixed integer optimization and Gurobi solver
    * Forward stepwise selection : QR decomposition
    * The lasso (and the relaxed lasso) : Coordinate descent implemented in  `glmnet`
2. Reproduce a table and visualizations such as Table 1 and Figure 5 to compare the four methods (the relaxed lasso in addition to the three methods) in terms of computation time and accuracy.
    * Table 1 : The time taken by each method to compute solutions.
    * Figure 5 : The accuracy metrics as functions of the signal to noise ratio (SNR) level.
3. Understand the `R` source codes written by Hastie et al. and write our own `julia` code.

## Reference

1. Hastie Trevor, Robert Tibshirani, and Ryan Tibshirani (2020). “Best subset, forward stepwise or lasso? Analysis and recommendations based on extensive comparisons.” Statistical Science 35(4), 579-592.
2. Bertsimas Dimitris, Angela King, and Rahul Mazumder (2016). “Best subset selection via a modern optimization lens.” Annals of Statistics 44(2), 813-852.
