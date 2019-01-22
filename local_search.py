import numpy as np
import numpy.matlib as mlb
import cvxpy as cvx
from itertools import combinations
from LinReg import LinReg
from sample_models import sample_models

def local_search(objective, inputs):
# LOCAL_SEARCH: Function runs binary optimization by searching over the neighborhood
# of single model flips at each iteration

    # Extract inputs
    nVars  = inputs['n_vars'] #n_vars = inputs.n_vars;
    nEval = inputs['evalBudget'] #n_iter = inputs.evalBudget;

    # Generate initial condition and evaluate objective
    model = sample_models(1, nVars)#;
    model_val = objective(model)#;

    # determine the total number of iterations
    nIter = int(np.ceil(nEval / nVars))#;

    # Setup cells to store model, objective, and runtime
    model_iter = np.zeros((nIter, nVars))#;
    obj_iter = np.zeros(nIter)#;
    #time_iter  = zeros(nIter,1);

    # Declare counter
    counter = 0#;

    for i in range(nIter):

        #ls_iter = tic;

        # Setup vector to store new objective values and difference
        new_obj  = np.zeros(nVars)#;
        diff_obj = np.zeros(nVars)#;

        for j in range(nVars):

            # Setup new_model with one flipped variable
            new_model = model#;
            new_model[0, j] = 1 - new_model[0, j]#;

            # Evaluate objective
            new_obj[j]  = objective(new_model)#;
            diff_obj[j] = model_val - new_obj[j]#;

        #end

        # Check if diff_obj is positive - improvement can be made
        if np.any(diff_obj > 0):

            # Choose optimal index to flip
            opt_idx = np.argmax(diff_obj)#;
            model[0, opt_idx] = 1 - model[0, opt_idx]#;
            model_val = new_obj[opt_idx]#;

        #end

        # Save models, model_obj, and runtime
        model_iter[counter,:] = model
        obj_iter[counter]  = model_val
        #time_iter(counter) = toc(ls_iter)
        
        # Update counter
        counter = counter + 1

    #end

    # extend results
    model_iter_new = np.zeros((nVars*nIter, nVars))
    #time_iter_new  = zeros(nVars*nIter, 1)
    for i in range(nIter):
        idx = np.arange(nVars * i, nVars * (i+1))
        model_iter_new[idx,:] = mlb.repmat(model_iter[i,:], nVars, 1)
        #time_iter_new(idx) = interp1([0,1],[0,time_iter(i)],linspace(1/nVars,1,nVars));
    #end
    obj_iter = np.reshape(mlb.repmat(obj_iter, 1, nVars).T, (nVars*nIter, 1))

    # save outputs
    #output = struct;
    #output.objVals  = obj_iter; 
    #output.optModel = model_iter_new;
    #output.runTime  = time_iter_new;

    #end
    return (model_iter_new, obj_iter)