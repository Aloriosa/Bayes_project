# Author: Ricardo Baptista and Matthias Poloczek
# Date:   June 2018
#
# See LICENSE.md for copyright information
#
import numpy as np
def KL_divergence_ising(Theta_P, moments, x):
# KL_divergence_ising: Function evaluates the KL divergence objective
# for Ising Models

    n_vars = Theta_P.shape[0]

    # Generate all binary vectors
    bin_vals = []
    for i in range(2 ** n_vars):
        tmp = np.array(map(int, list(("{0:0" + str(n_vars) + "b}").format(i))))
        tmp[np.where(tmp == 0)] = -1
        bin_vals.append(list(tmp))
    #bin_vals = "{0:b}".format(0:2 ** n_vars-1)-'0';
    #bin_vals[bin_vals == 0] = -1;
    bin_vals = np.array(bin_vals)
    n_vectors = bin_vals.shape[0]
    
    
    # Compute normalizing constant for P
    P_vals = np.zeros((n_vectors,1))
    for i in range(n_vectors):
        P_vals[i] = np.exp(bin_vals[i,:].dot(Theta_P).dot(bin_vals[i,:].T))

    Zp = P_vals.sum()

    # Run computation for each x
    n_xvals = x.shape[0]
    KL = np.zeros((n_xvals,1))

    for j in range(n_xvals):

        # Apply elementwise masking to Theta
        Theta_Q = np.tril(Theta_P,-1)
        nnz_Q   = np.where(Theta_Q > 0)
        Theta_Q[nnz_Q] = np.multiply(Theta_Q[nnz_Q], x[nnz_Q].T)#x[j,:].T)
        Theta_Q = Theta_Q + Theta_Q.T

        # Compute normalizing constant for Q
        Q_vals = np.zeros((n_vectors,1))
        for i in range(n_vectors):
            Q_vals[i] = np.exp(bin_vals[i,:].dot(Theta_Q).dot(bin_vals[i,:].T))

        Zq = Q_vals.sum()

        # compute KL
        KL[j] = np.multiply(Theta_P - Theta_Q, moments).sum() + np.log(Zq) - np.log(Zp)


    return KL