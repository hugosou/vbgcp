# import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from examples.toydata import build_toydaset
from vbgcp.vb_gcp import VBGCPTensor
from vbgcp.components import FitParams
from vbgcp.utils import (plot_factors, expand_factors, get_similarity, reorder_models)

np.random.seed(1)
params = build_toydaset(add_offset=1)

factors_true = params['factors']
observed_tensor = params['observed_tensor']
neuron_groups = params['neurons_groups']

observed_data = np.random.rand(*observed_tensor.shape) > 0.5
observed_tensor = observed_tensor * observed_data

tensor_rank = 6

fit_params = FitParams(observed_tensor.shape, tensor_rank,
                       observed_data=observed_data, fit_offset_dim=[1, 0, 1, 0, 0],
                       shared_precision_dim=[0, 1, 1, 1, 1], shared_precision_mode=0,
                       neuron_groups=neuron_groups, ite_max=2)

vbgcp = VBGCPTensor(observed_tensor.shape, tensor_rank, shape_param=120, fit_params=fit_params)
vbgcp.variational_inference(observed_tensor)

#%%

models = [expand_factors(factors_true, 6), vbgcp.posteriors.factors_mean]
smlty, _,_ = get_similarity(models)
print(smlty)
models = reorder_models(models)


plt.figure()
plot_factors(models[0])
plot_factors(models[1], color='m')

#%%
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(vbgcp.shape_param_tot, color='k')
plt.plot(np.ones(len(vbgcp.shape_param_tot))*params['model']['shape'], color='m')

plt.title('Shape Param')
plt.xlabel('Iterations')

plt.subplot(1, 2, 2)
plt.plot(vbgcp.loss_tot, color='k')
plt.xlabel('Iterations')
plt.title('Loss')

plt.figure()
plot_factors(vbgcp.posteriors.factors_mean)

plt.figure()
plot_factors(params['factors'])

# TODO: check offset + add similarity + check initialization + plot and normalize cariance, check missing data
# TODO: when checking offset, also chek init of the offset
# TODO: add disppct

#%%


#%%
# TODO also reorder the variance ?
# TODO: add a checker for size and rank. reorder and normalize cp. Then use first CPs if rank different from ref
# TODO: check that normalise cp with Normdim works !


# TODO also reorder the variance ?
# TODO: add a checker for size and rank. reorder and normalize cp. Then use first CPs if rank different from ref
# TODO: check that normalise cp with Normdim works !

# TODO: check offset + add similarity + check initialization + plot and normalize cariance, check missing data
# TODO: when checking offset, also chek init of the offset
# TODO: add disppct



