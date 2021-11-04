import numpy as np
from vbgcp.utils import get_dim_factors, rand_factors, check_rank_factors, eyes_precisions, \
    zeros_factors, compact_to_full_offset


class FitParams:
    def __init__(self, tensor_shape, tensor_rank, observed_data=1, fit_offset_dim=None, fit_factors_dim=None,
                 fit_shape_param=True, shape_update='MM-G', shared_precision_dim=None, shared_precision_mode=None,
                 neuron_groups=None, ite_max=1000, disppct=1):
        """ Variational Inference Parameters
        :param tensor_shape: Shape of the observed tensor
        :param tensor_rank: Assumed Rank of the observed tensor
        :param observed_data: Missing Data
        :param fit_offset_dim: Offset dimensions to fit
        :param fit_factors_dim: Factors dimensions to fit
        :param fit_shape_param: Do fit shape parameter
        :param shape_update: Method to estimate PG-KL in ELBO
        :param shared_precision_dim: Share precision (ARD)
        :param shared_precision_mode: Share precision (Mode-wise)
        :param neuron_groups: Define groups for this mode
        :param ite_max: Variational Expectation Maximization iterations
        """

        # Dimensions of the problem
        self.rank = tensor_rank
        self.shape = list(tensor_shape)

        # Missing Data (Default: None)
        self.observed_data = observed_data
        if np.all(observed_data) == 1:
            self.observed_data = 1

        # Fit offset (Default: No)
        if (fit_offset_dim is None) or np.all(fit_offset_dim == np.zeros(len(self.shape))):
            self.fit_offset = False
            self.fit_offset_dim = np.zeros(len(self.shape), dtype=bool)
        else:
            assert len(fit_offset_dim) == len(self.shape), 'Incorrect offset dimensions'
            self.fit_offset = True
            self.fit_offset_dim = fit_offset_dim

        # Shape parameter and fitting method
        self.fit_shape_param = fit_shape_param
        self.shape_update = shape_update

        # Fit factors
        if fit_factors_dim is None:
            fit_factors_dim = np.ones(len(self.shape), dtype=bool)
        else:
            assert len(fit_factors_dim) == len(self.shape), 'Incorrect factors dimensions'
        self.fit_factors_dim = fit_factors_dim

        # ARD-type procedures
        if shared_precision_dim is None:
            shared_precision_dim = np.zeros(len(self.shape), dtype=bool)
        else:
            assert len(shared_precision_dim) == len(self.shape), 'Incorrect shared precision dimensions'
        self.shared_precision_dim = shared_precision_dim

        if not(shared_precision_mode is None):
            assert not(neuron_groups is None), 'No mode groups provided'
            assert (self.shape[shared_precision_mode] == neuron_groups.shape[0]), 'Groups and Mode dimension mismatch'


        self.neuron_groups = neuron_groups
        self.shared_precision_mode = shared_precision_mode

        self.ite_max = ite_max
        self.disppct = disppct


class VBGCPPriors:
    def __init__(self, fit_params: FitParams, factors_mean=None, factors_precision=None,
                 offset_mean=None, offset_precision=None, a_shared=100, b_shared=1, a_mode=100, b_mode=1):
        """ Priors for Variational Inference """
        """VBCPPrior"""

        # Factors: Mean
        if factors_mean is None:
            factors_mean = zeros_factors(fit_params.shape, fit_params.rank)
        self.factors_mean = factors_mean

        # Factors: Precision
        if factors_precision is None:
            factors_precision = eyes_precisions(fit_params.shape, fit_params.rank, weight=0.01)
        self.factors_precision = factors_precision

        # Offset: Mean (compact form)
        offset_shape = [fit_params.shape[ii] for ii in np.where(fit_params.fit_offset_dim)[0]]
        if offset_mean is None:
            offset_mean = np.zeros(offset_shape)
        self.offset_mean = offset_mean

        # Offset: Precision (compact form)
        if offset_precision is None:
            offset_precision = 1e-5 * np.ones(offset_shape)
        self.offset_precision = offset_precision

        # Precision priors parameters
        self.a_shared = a_shared
        self.b_shared = b_shared
        self.a_mode = a_mode
        self.b_mode = b_mode

        self.check_priors(fit_params)

    def check_priors(self, fit_params):

        assert get_dim_factors(self.factors_mean) == fit_params.shape, \
            'Incorrect prior factors dimensions'

        assert check_rank_factors(self.factors_mean) == fit_params.rank, \
            'Incorrect prior factors rank'

        assert get_dim_factors(self.factors_precision) == fit_params.shape, \
            'Incorrect prior precision dimensions'

        assert check_rank_factors(self.factors_precision) == fit_params.rank ** 2, \
            'Incorrect prior precision rank'

        offset_shape = [fit_params.shape[ii] for ii in np.where(fit_params.fit_offset_dim)[0]]
        assert offset_shape == list(self.offset_mean.shape) or \
        (offset_shape == [] and np.all(self.offset_mean.shape) == 0), \
            'Incorrect offset mean dimensions'

        assert offset_shape == list(self.offset_precision.shape)or \
        (offset_shape == [] and np.all(self.offset_mean.shape) == 0), \
            'Incorrect offset precision dimensions'


class VBGCPPosteriors:
    def __init__(self, fit_params: FitParams, factors_mean=None, factors_variance=None,
                 offset_mean=None, offset_variance=None, latent_mean=None):

        self.latent_mean = latent_mean

        # Factors: Mean
        if factors_mean is None:
            factors_mean = rand_factors(fit_params.shape, fit_params.rank, weight=1)
        self.factors_mean = factors_mean

        # Factors: Variance
        if factors_variance is None:
            factors_variance = _init_factors_posterior_variance(fit_params.shape, fit_params.rank)
        self.factors_variance = factors_variance

        # Offset: Mean (same size as observed tensor)
        if offset_mean is None:
            if fit_params.fit_offset:
                offset_mean = _init_offset_posterior(fit_params.shape,
                                                     fit_params.fit_offset_dim,
                                                     fit_params.observed_data)
            else:
                offset_mean = np.zeros(fit_params.shape)
        self.offset_mean = offset_mean

        # Offset: Variance (same size as observed tensor)
        if offset_variance is None:
            if fit_params.fit_offset:
                offset_variance = _init_offset_posterior(fit_params.shape,
                                                         fit_params.fit_offset_dim,
                                                         fit_params.observed_data)
            else:
                offset_variance = np.zeros(fit_params.shape)
        self.offset_variance = offset_variance
        self.check_posteriors(fit_params)

    def check_posteriors(self, fit_params):

        assert get_dim_factors(self.factors_mean) == fit_params.shape, \
            'Incorrect posterior factors dimensions'

        assert check_rank_factors(self.factors_mean) == fit_params.rank, \
            'Incorrect posterior factors rank'

        assert get_dim_factors(self.factors_variance) == fit_params.shape, \
            'Incorrect posterior factor precision dimensions'

        assert check_rank_factors(self.factors_variance) == fit_params.rank ** 2, \
            'Incorrect posterior factor precision rank'

        assert fit_params.shape == list(self.offset_mean.shape), \
            'Incorrect posterior offset mean dimensions'

        assert fit_params.shape == list(self.offset_variance.shape), \
            'Incorrect posterior offset mean dimensions'


def _init_offset_posterior(tensor_shape, fit_offset_dim, observed_data):

    # # Shape of the compact form
    # offset_shape = [tensor_shape[ii] for ii in np.where(fit_offset_dim)[0]]
    #
    # # Dimensions along which offset can vary
    # to_ones = np.where(fit_offset_dim)[0]
    #
    # # Dimensions along which offset is tiled
    # to_repeat = np.where(1 - np.array(fit_offset_dim))[0]
    #
    # # For reordering
    # tmp_permute = np.concatenate((to_repeat[:], to_ones[:]))
    # inv_permute = np.arange(tmp_permute.size)
    # for i in np.arange(tmp_permute.size):
    #     inv_permute[tmp_permute[i]] = i
    #
    # # Compact Init (could use something else than zeros)
    # tmp = np.zeros(offset_shape)
    #
    # # Tile offset
    # tmp = np.tile(tmp, np.concatenate(([tensor_shape[i] for i in to_repeat], [1 for _ in to_ones])))
    #
    # # Store
    # offset = np.transpose(tmp, inv_permute) * observed_data

    # Shape of the compact form
    offset_shape = [tensor_shape[ii] for ii in np.where(fit_offset_dim)[0]]

    # Compact Init (could use something else than zeros)
    compact_offset = np.zeros(offset_shape)

    # Full Offset
    offset = compact_to_full_offset(compact_offset, tensor_shape, fit_offset_dim) * observed_data

    return offset


def _init_factors_posterior_variance(tensor_shape, tensor_rank, weight=0.1):
    factors_variance = eyes_precisions(tensor_shape, tensor_rank, weight=0.1)

    for dim_ext in range(len(tensor_shape)):
        for dim_int in range(tensor_shape[dim_ext]):
            tmp = weight*np.random.normal(size=(tensor_rank, tensor_rank))
            tmp = np.transpose(tmp) @ tmp
            factors_variance[dim_ext][dim_int   , :] = np.reshape(tmp, (-1))

    return factors_variance


