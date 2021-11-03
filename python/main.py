# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # %% The only difference!

    import numpy as np
    # import jax.numpy as jnp
    import matplotlib.pyplot as plt

    from toydata import build_toydaset
    from vb_gcp import VBGCPTensor
    from components import FitParams
    from utils import plot_factors

    np.random.seed(1)
    params = build_toydaset(add_offset=1, add_missing=1)

    factors_true = params['factors']
    observed_tensor = params['observed_tensor']
    observed_data = params['observed_data']

    tensor_rank = 6
    fit_params = FitParams(observed_tensor.shape, tensor_rank,
                           observed_data=params['observed_data'], fit_offset_dim=[1, 0, 1, 0, 0],
                           shared_precision_dim=[0, 1, 1, 1, 1], shared_precision_mode=0,
                           neuron_groups=params['neurons_groups'], ite_max=4000)

    vbgcp = VBGCPTensor(observed_tensor.shape, tensor_rank, shape_param=120, fit_params=fit_params)
    vbgcp.variational_inference(observed_tensor)

    plot_factors(vbgcp.posteriors.factors_mean)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

