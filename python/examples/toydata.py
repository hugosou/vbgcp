import numpy as np
from vbgcp.utils import cp_to_tensor


def build_toydaset(tshape=np.array([100, 70, 3, 4, 5]), add_offset=1, add_missing=0, model='negative_binomial'):

    # Fixed tensor rank = 4
    rank = 4

    # Time dynamics
    FT = np.zeros((tshape[1], rank))
    FT[:, 0] = np.sin(np.linspace(0, 2*np.pi,tshape[1] ))-1
    FT[:, 1] = (1-np.cos(np.linspace(0, 2*np.pi, tshape[1])))
    FT[:, 2] = np.abs(np.sin(np.linspace(0, 2*np.pi, tshape[1])))**2
    FT[:, 3] = np.abs(np.sin(np.linspace(0, 2*np.pi, tshape[1])))**2
    FT = FT - np.mean(FT, axis=0)

    # Neuron loadings
    FN = np.random.normal(loc=0.0, scale=2.0, size=(tshape[0], rank))


    # Neuron Groups
    neurons_groups = np.zeros((tshape[0], 3))
    neurons_groups[:, 0] = (1+np.arange(tshape[0]) < tshape[0]/3).astype(int)
    neurons_groups[:, 1] = (1+np.arange(tshape[0]) >= tshape[0]/3).astype(int)*(1+np.arange(tshape[0]) < 2*tshape[0]/3).astype(int)
    neurons_groups[:, 2] = (1+np.arange(tshape[0]) >= 2*tshape[0]/3).astype(int)

    # Assign Factors to group
    group_1_pc = np.sum(neurons_groups*[0, 1, 1], axis=1, keepdims=1)
    group_2_pc = np.sum(neurons_groups*[1, 0, 1], axis=1, keepdims=1)
    group_3_pc = np.sum(neurons_groups*[0, 1, 0], axis=1, keepdims=1)
    group_4_pc = np.sum(neurons_groups*[1, 1, 1], axis=1, keepdims=1)
    group_pc = np.concatenate((group_1_pc, group_2_pc, group_3_pc, group_4_pc), axis=1)

    # Final Neuron Loadings
    FN *= group_pc

    # Condition Factor
    FC = np.zeros((tshape[2], rank))
    FC[:, 0] = [1, 1, 1]
    FC[:, 1] = [0.2, 0, 1]
    FC[:, 2] = [1, 0, 1]
    FC[:, 3] = [0, 1, 0]

    # Experiment Factor
    FE = np.ones((tshape[3], rank)) + np.random.normal(size=(tshape[3], rank))

    # Trial Factor
    FK = np.zeros((tshape[4], rank))
    FK[:, 0] = np.linspace(0, 1, tshape[4])
    FK[:, 1] = np.linspace(1, 0, tshape[4])
    FK[:, 2] = np.linspace(0, 1, tshape[4])**2
    FK[:, 3] = np.linspace(1, 1, tshape[4])

    # Full Factors
    factors_true = [FN, FT, FC, FE, 0.1*FK]

    # Dynamics
    Wtrue = cp_to_tensor(factors_true)

    # Add an offset
    offset_dim = np.array([1, 0, 1, 0, 0])
    vtrue = np.random.rand(*(offset_dim*tshape + (1-offset_dim)))
    vtrue = 0.1*vtrue
    offset = add_offset*vtrue

    # Observation model
    if model == 'gaussian':
        noise = np.random.normal(scale=0.1, size=tshape)
        Xtrue = Wtrue
        Xobs = Xtrue + noise
        model_tot={'name':model, "scale":0.1}

    elif model == 'poisson':
        Xtrue = np.exp(Wtrue)
        Xobs = np.random.poisson(lam= Xtrue)
        model_tot = {'name': model}

    elif model == 'negative_binomial':
        ashape = 80 * np.ones(Wtrue.shape)
        aparam = 1/(1+np.exp(Wtrue))
        Xobs = np.random.negative_binomial(ashape,aparam)
        Xtrue = ashape * np.exp(aparam)
        model_tot = {'name': model, "shape": 80}


    # Model Missing Entries
    if add_missing:
        neuron_expt = (np.array([0, np.floor(tshape[0]/6), np.floor(tshape[0]/2), np.floor(3*tshape[0]/4), tshape[0] ])).astype(int)
        observed_data = np.zeros((tshape[0],tshape[3]))
        observed_data[neuron_expt[0]:neuron_expt[1],0] = 1
        observed_data[neuron_expt[1]:neuron_expt[2],1] = 1
        observed_data[neuron_expt[2]:neuron_expt[3],2] = 1
        observed_data[neuron_expt[3]:neuron_expt[4],3] = 1
        observed_data = np.transpose(np.tile(observed_data, (tshape[1], tshape[2], tshape[4], 1, 1)), axes=(3, 0, 1, 4, 2))
    else:
        observed_data = 1

    Xobs = Xobs * observed_data

    params = {
        "factors": factors_true,
        "offset": offset,
        "offset_dim":offset_dim,
        "neurons_groups":neurons_groups,
        "observed_data":observed_data,
        "observed_tensor":Xobs,
        "noiseless_tensor":Xtrue,
        "model":model_tot
    }

    return params