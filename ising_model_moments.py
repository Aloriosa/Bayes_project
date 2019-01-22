import numpy as np

def ising_model_moments(Q):
#Q - matrix
    n_vars = Q.shape[0]

    #Generate all binary vectors
    bin_vals = []
    for i in range(2 ** n_vars):
        tmp = np.array(map(int, list(("{0:0" + str(n_vars) + "b}").format(i))))
        tmp[np.where(tmp == 0)] = -1
        bin_vals.append(list(tmp))
    #bin_vals = "{0:b}".format(0:2 ** n_vars-1)-'0';
    #bin_vals[bin_vals == 0] = -1;
    bin_vals = np.array(bin_vals)
    n_vectors = bin_vals.shape[0]

    # Compute values of PDF
    pdf_vals = np.zeros((n_vectors,1))
    for i in range(n_vectors):
        pdf_vals[i] = np.exp(bin_vals[i,:].dot(Q).dot(bin_vals[i,:].T))


    # Compute normalizing constant
    norm_const = sum(pdf_vals)

    # Generate matrix to store moments
    ising_mom = np.zeros((n_vars,n_vars))

    # Compute second moment for each pair of values
    for i in range(n_vars):
        for j in range(n_vars):
            bin_pair = np.multiply(bin_vals[:,i], bin_vals[:,j])
            ising_mom[i,j] = np.sum(np.multiply(bin_pair, pdf_vals)) / norm_const

    return ising_mom