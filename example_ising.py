import numpy as np
import matplotlib.pyplot as plt
from BOCS import BOCS
from ising_model_moments import ising_model_moments
from rand_ising_grid import rand_ising_grid
from KL_divergence_ising import KL_divergence_ising
#from quad_mat import quad_mat
from sample_models import sample_models

inputs = {}
#inputs['n_nodes'] = 16
inputs['n_vars']     = 12
inputs['evalBudget'] = 100
inputs['n_init']     = 20
inputs['lambda']     = 1e-4

# Generate random 3x3 graphical model
Theta   = rand_ising_grid(9)
Moments = ising_model_moments(Theta)

# Save objective function and regularization term

inputs['model']    = lambda x: KL_divergence_ising(Theta, Moments, x) # compute x^TQx row-wise
inputs['penalty']  = lambda x: inputs['lambda']*np.sum(x,axis=1)



# Generate initial samples for statistical models
inputs['x_vals']   = sample_models(inputs['n_init'], inputs['n_vars'])
inputs['y_vals']   = inputs['model'](inputs['x_vals'])

# Run BOCS-SA and BOCS-SDP (order 2)
(BOCS_SA_model, BOCS_SA_obj)   = BOCS(inputs.copy(), 2, 'SA')
(BOCS_SDP_model, BOCS_SDP_obj) = BOCS(inputs.copy(), 2, 'SDP-l1')

# Compute optimal value found by BOCS
iter_t = np.arange(BOCS_SA_obj.size)
BOCS_SA_opt  = np.minimum.accumulate(BOCS_SA_obj)
BOCS_SDP_opt = np.minimum.accumulate(BOCS_SDP_obj)

# Compute minimum of objective function
n_models = 2**inputs['n_vars']
x_vals = np.zeros((n_models, inputs['n_vars']))
str_format = '{0:0' + str(inputs['n_vars']) + 'b}'
for i in range(n_models):
	model = str_format.format(i)
	x_vals[i,:] = np.array([int(b) for b in model])
f_vals = inputs['model'](x_vals) + inputs['penalty'](x_vals)
opt_f  = np.min(f_vals)

# Plot results
fig = plt.figure()
ax  = fig.add_subplot(1,1,1)
ax.plot(iter_t, np.abs(BOCS_SA_opt - opt_f), color='r', label='BOCS-SA')
ax.plot(iter_t, np.abs(BOCS_SDP_opt - opt_f), color='b', label='BOCS-SDP')
ax.set_yscale('log')
ax.set_xlabel('$t$')
ax.set_ylabel('Best $f(x)$')
ax.legend()
fig.savefig('BOCS_simpleregret_ising.pdf')
plt.close(fig)

# -- END OF FILE --