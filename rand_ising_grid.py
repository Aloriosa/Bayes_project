
import numpy as np

def rand_ising_grid(n_vars):
# RAND_ISING: Function generates a random interaction matrix
# for an Ising Model on a grid graph with n_vars total variables

# Check that n_side is an integer
    n_side = np.int(np.round(np.sqrt(n_vars)))
    if n_side**2 != n_vars:
        print('Number of nodes is not square')

    Q = np.zeros((n_vars, n_vars))
    # Connect nodes horizontally
    for i in range(n_side):
        for j in range(n_side - 1):
            # Determine node idx
            node = i*n_side + j

            Q[node,node+1] = 4.95*np.random.rand() + 0.05; #0.95*rand() + 0.05;
            Q[node+1,node] = Q[node,node+1]


    # Connect nodes vertically
    for i in range(n_side-1):
        for j in range(n_side):

             # Determine node idx
            node = i*n_side + j

            Q[node, node+n_side] = 4.95*np.random.rand() + 0.05 #0.95*rand() + 0.05;
            Q[node+n_side,node] = Q[node,node+n_side]


    # Apply random sign flips to Q
    rand_sign = np.tril((np.random.rand(n_vars,n_vars) > 0.5)*2 - 1,-1)
    rand_sign = rand_sign + rand_sign.T
    Q = np.multiply(rand_sign, Q)

    return Q