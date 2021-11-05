import numpy as np
from vb_gcp import VBGCPTensor
from utils import *
from components import *
import scipy.io


# Tensor Rank
rank = 10

# Tensor Shape
tshape = [10, 5, 2]

# Random Factors
factor_test = rand_factors(tshape, rank)


#%% Test: tensor reconstruction
# Khatri-Rao Products
tensor_test = cp_to_tensor(factor_test)

# Brute force tensor reconstruction
tensor_try = np.zeros(get_dim_factors(factor_test))
for rr in range(rank):
    for ii in range(get_dim_factors(factor_test)[0]):
        for jj in range(get_dim_factors(factor_test)[1]):
            for kk in range(get_dim_factors(factor_test)[2]):
              tensor_try[ii,jj,kk] += factor_test[0][ii,rr]*factor_test[1][jj,rr]*factor_test[2][kk,rr]

assert np.sum(np.abs(vectorize(tensor_try-tensor_test))) < 1e-10, "Incorect tensor reconstruction"
print('pass')

#%% Test: Variance reconstruction
MMt = get_MMt(factor_test)
for ii in range(len(factor_test)):
    Mi = factor_test[ii]
    MMi = np.zeros((Mi.shape[0], Mi.shape[1]**2))

    for jj in range(Mi.shape[0]):
        Mij = Mi[jj, :]
        MMi[jj, :] = np.reshape(np.expand_dims(Mij, axis=1) @ np.expand_dims(Mij, axis=0), (1, Mi.shape[1]**2))

    assert np.all(MMi - MMt[ii] == 0), str(ii)
print('pass')


#%% Compare Python and Matlab updates

def mat_to_vbgcp(mat):
    """Load matlab VBGCP structures and 'translate' them in python"""

    Xobs = mat['Xobs']

    vi_param = mat['vi_param']
    ite_max = np.squeeze(vi_param['ite_max'][0, 0])
    observed_data = vi_param['observed_data'][0,0]
    fit_offset_dim = np.squeeze(vi_param['fit_offset_dim'][0,0])
    shared_precision_dim = np.squeeze(vi_param['shared_precision_dim'][0,0])
    dim_neuron = np.squeeze(vi_param['dim_neuron'][0,0][0, 0]-1)
    neurons_groups = vi_param['neurons_groups'][0,0]
    update_CP_dim = np.squeeze(vi_param['update_CP_dim'][0,0])
    shape_update = vi_param['shape_update'][0,0][0]

    vi_var = mat['vi_var_with_ard']
    shape_param = vi_var['shape'][0,0][0]

    factors_prior_mean = [facti for facti in vi_var['CP_prior_mean'][0, 0][0, :]]
    factors_posterior_mean = [facti for facti in vi_var['CP_mean'][0, 0][0, :]]

    factors_prior_precision = [facti for facti in vi_var['CP_prior_precision'][0, 0][0, :]]
    factors_posterior_variance = [facti for facti in vi_var['CP_variance'][0, 0][0, :]]

    offset_prior_mean = vi_var['offset_prior_mean'][0, 0]
    offset_prior_precision = vi_var['offset_prior_precision'][0, 0]

    prior_a_mode = vi_var['prior_a_mode'][0, 0][0, 0]
    prior_b_mode = vi_var['prior_b_mode'][0, 0][0, 0]

    prior_a_shared = vi_var['prior_a_shared'][0, 0][0, 0]
    prior_b_shared = vi_var['prior_b_shared'][0, 0][0, 0]

    offset_posterior_mean = vi_var['offset_mean'][0, 0]
    offset_posterior_variance = vi_var['offset_variance'][0, 0]

    tensor_m1 = vi_var['tensor_mean'][0, 0]
    tensor_m2 = vi_var['tensor2_mean'][0, 0]

    latent_posterior_mean = vi_var['latent_mean'][0, 0]

    tensor_m10 = cp_to_tensor(factors_posterior_mean)
    tensor_m20 = cp_to_tensor(get_AAt(factors_posterior_mean, factors_posterior_variance))

    assert np.sum(np.abs(tensor_m1-tensor_m10)) < 1e-10
    assert np.sum(np.abs(tensor_m2-tensor_m20)) < 1e-10

    FE = vi_var['FE'][0, 0][0,0]

    tensor_shape = Xobs.shape
    tensor_rank = check_rank_factors(factors_posterior_mean)

    fit_params = FitParams(tensor_shape, tensor_rank, observed_data=observed_data, fit_offset_dim=fit_offset_dim,
                      shape_update=shape_update, shared_precision_dim=shared_precision_dim,
                      shared_precision_mode=dim_neuron, neuron_groups=neurons_groups, ite_max=ite_max)

    priors = VBGCPPriors(fit_params=fit_params,
                         factors_mean=factors_prior_mean, factors_precision=factors_prior_precision,
                         offset_mean=offset_prior_mean, offset_precision=offset_prior_precision,
                         a_shared=prior_a_shared, b_shared=prior_b_shared,
                         a_mode=prior_a_mode, b_mode=prior_b_mode)

    posteriors = VBGCPPosteriors(fit_params=fit_params,
                                 factors_mean=factors_posterior_mean, factors_variance=factors_posterior_variance,
                                 offset_mean=offset_posterior_mean, offset_variance=offset_posterior_variance,
                                 latent_mean=latent_posterior_mean)

    vcbgcp = VBGCPTensor(tensor_shape, tensor_rank, shape_param=shape_param,
                         fit_params=fit_params, priors=priors, posteriors=posteriors)

    vcbgcp.posteriors.tensor_m1 = tensor_m1
    vcbgcp.posteriors.tensor_m2 = tensor_m2

    vcbgcp.loss_tot = []
    vcbgcp.shape_param_tot = []

    CPtrue =  [facti for facti in mat['CP'][0]]

    return vcbgcp, Xobs, CPtrue


import scipy.io
from vbgcp.components import *
from vbgcp.utils import *
from vbgcp.vb_gcp import VBGCPTensor

# Raw matlab structure
mat = scipy.io.loadmat('/home/sou/Documents/PYTHON/tests_matlab/for_testing.mat')
vbgcp, Xobs, CP = mat_to_vbgcp(mat)

# Updated matlab structure
mat2 = scipy.io.loadmat('/home/sou/Documents/PYTHON/tests_matlab/for_testing2.mat')
vbgcp2, _, _ = mat_to_vbgcp(mat2)

# Update Raw struture
vbgcp.fit_params.ite_max = 100


#vbgcp.variational_inference(Xobs)

#vbgcp._update_offset(Xobs)

observed_tensor = Xobs
#%%

offset = mat['offset'][:,0,:,0,0]
offset_fit = vbgcp.posteriors.offset_mean[:,0,:,0,0]

plt.figure()
for ii in np.arange(offset.shape[1]):
    plt.subplot(offset.shape[1],1,ii+1)
    plt.plot(offset[:,ii])
    plt.plot(offset_fit[:, ii])



#%%




factors_post_mean_diff = np.max([np.max(np.abs( vbgcp.posteriors.factors_mean[ii] - vbgcp2.posteriors.factors_mean[ii]))
              for ii in range(len(observed_tensor.shape))])

factors_post_variance_diff = np.max([np.max(np.abs(vbgcp.posteriors.factors_variance[ii] - vbgcp2.posteriors.factors_variance[ii]))
              for ii in range(len(observed_tensor.shape))])

latent_diff = np.max(np.abs(vbgcp.posteriors.latent_mean - vbgcp2.posteriors.latent_mean))

factors_prior_mean_diff = np.max([np.max(np.abs(vbgcp.priors.factors_mean[ii] - vbgcp2.priors.factors_mean[ii]))
              for ii in range(len(observed_tensor.shape))])

factors_prior_precision_diff = np.max([np.max(np.abs(vbgcp.priors.factors_precision[ii] - vbgcp2.priors.factors_precision[ii]))
              for ii in range(len(observed_tensor.shape))])

offset_mean_diff = np.max(np.abs(vbgcp.posteriors.offset_mean - vbgcp2.posteriors.offset_mean))

offset_variance_diff = np.max(np.abs(vbgcp.posteriors.offset_variance - vbgcp2.posteriors.offset_variance))

shape_diff = np.abs(vbgcp2.shape_param-vbgcp.shape_param)

print("Factors post mean diff: " + str(factors_post_mean_diff))
print("Factors variance diff: " + str(factors_post_variance_diff))
print("Latent Diff: " + str(latent_diff))
print("Factors prior mean diff: " + str(factors_prior_mean_diff))
print("Factors precision diff: " + str(factors_prior_precision_diff))
print("Offset mean diff: " + str(offset_mean_diff))
print("Offset vari diff: "+ str(offset_variance_diff))
print("Shape diff:" + str(shape_diff))


factors_post_mean_diff
factors_post_variance_diff
latent_diff
factors_prior_mean_diff
factors_prior_precision_diff
offset_mean_diff
offset_variance_diff
shape_diff


#%%

models = [expand_factors(CP, 6), vbgcp.posteriors.factors_mean, vbgcp2.posteriors.factors_mean]
get_similarity(models, ref_model=0)


#%%

OT = observed_tensor
neurons_groups = vbgcp.fit_params.neuron_groups
OD = vbgcp.fit_params.observed_data

tensor_rank = 6
fit_params = FitParams(observed_tensor.shape, 6, observed_data=OD, fit_offset_dim=[1, 0, 1, 0, 0],
                       shared_precision_dim=[0, 1, 1, 1, 1], shared_precision_mode=0,
                       neuron_groups=neurons_groups, ite_max=200)
vv = VBGCPTensor(observed_tensor.shape, 6, shape_param=85, fit_params=fit_params)
vv.variational_inference(OT)


