# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.





# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # %% The only difference!

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
                           neuron_groups=neuron_groups, ite_max=100)

    vbgcp = VBGCPTensor(observed_tensor.shape, tensor_rank, shape_param=80, fit_params=fit_params)
    vbgcp.variational_inference(observed_tensor)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

