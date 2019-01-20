
# coding: utf-8

# In[ ]:


import numpy as np
import cvxpy as cvx
from itertools import combinations
from LinReg import LinReg
from sample_models import sample_models

def random_samp(objective, inputs):
    # RANDOM_SAMP: Function generates random samples to find the minimum of an 
    # objective function starting from the specified initial condition


    # Extract n_vars
    n_vars  = inputs['n_vars'] #n_vars = inputs.n_vars;
    n_iter = inputs['evalBudget'] #n_iter = inputs.evalBudget;

    # Generate initial condition and evaluate objective
    model = sample_models(1,n_vars)#;
    model_val = objective(model)#;

    # Setup cells to store model, objective, and runtime
    model_iter = np.zeros((n_iter, n_vars)) #model_iter = np.zeros(n_iter, n_vars);
    obj_iter   = np.zeros(n_iter) #obj_iter   = zeros(n_iter,1);
    #time_iter  = zeros(n_iter,1);

    # Declare counter
    counter = 0 #;

    for t in range(n_iter): #for i=1:n_iter

        
        #rand_iter = tic;

        # Sample random model and evaluate objective
        new_model = sample_models(1, n_vars)#;
        new_model_val = objective(new_model)#;

        # If model is better, update model, model_val 
        if new_model_val < model_val:
            model = new_model#;
            model_val = new_model_val#;
        #end

        # Save models, model_obj, and runtime
        model_iter[counter,:] = model #model_iter(counter,:) = model;
        obj_iter[counter]  = model_val #obj_iter(counter)  = model_val;
        #time_iter(counter) = toc(rand_iter);
        
        # Update counter
        counter = counter + 1#;
    #end

    # save outputs
    #output = struct;
    #output.objVals  = obj_iter; 
    #output.optModel = model_iter;
    #output.runTime  = time_iter;

    #end
    return (model_iter, obj_iter)

