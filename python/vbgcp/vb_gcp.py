import numpy as np
from vbgcp.components import FitParams, VBGCPPriors, VBGCPPosteriors
from vbgcp.utils import cp_to_tensor, get_AAt, pg_moment
from scipy.special import gammaln, psi
import scipy.optimize

import vbgcp._variational_updates


class VBGCPTensor(vbgcp._variational_updates.Mixin):
    def __init__(self, tensor_shape, tensor_rank, shape_param=None, fit_params=None, priors=None, posteriors=None):
        """
        Bayesian tensor CP decomposition of count data.
        Inference via approx. Variational Inference and Polya-Gamma augmentation.
        For a complete description, see Soulat et al. (2021).

        :param tensor_shape: Dimensions of the problem
        :param tensor_rank: Assumed rank of the decomposed tensor
        :param shape_param: Float, Shape Parameter of the generative model
        :param fit_params: FITParams, Variational Inference Parameters
        :param priors: VBGCPPriors, Priors for the model variables
        :param posteriors: VBGCPPosteriors, Current posterior estimates
        """
        # Dimensions of the problem
        self.rank = tensor_rank
        self.shape = list(tensor_shape)

        if fit_params is None:
            fit_params = FitParams(tensor_shape, tensor_rank)

        if priors is None:
            priors = VBGCPPriors(fit_params)

        if posteriors is None:
            posteriors = VBGCPPosteriors(fit_params)

        if not(shape_param is None):
            shape_param = float(shape_param)

        # Model
        self.priors = priors
        self.posteriors = posteriors
        self.fit_params = fit_params
        self.shape_param = shape_param

    def _init_inference(self, observed_tensor):
        """ Initialize Tensor Decomposition and Variational Inference Parameters """

        # TODO: implement sparse block structure detection

        # Check Priors and Posteriors shapes
        self.priors.check_priors(self.fit_params)
        self.posteriors.check_posteriors(self.fit_params)

        # Init Shape Param
        if self.shape_param is None:
            # self.shape_param = 100
            if not (np.all(self.fit_params.observed_data == 1)):
                self.shape_param = np.mean(observed_tensor[np.where(self.fit_params.observed_data)])
            else:
                self.shape_param = np.mean(observed_tensor)

        # Fasten element wise operations
        if np.all(self.fit_params.observed_data) == 1:
            self.observed_data = 1

        # Init 1st and 2nd moments of reconstructed tensor
        factors_mean = self.posteriors.factors_mean
        factors_variance = self.posteriors.factors_variance

        # 1st moment of reconstructed tensor <[|..|]>
        self.posteriors.tensor_m1 = cp_to_tensor(factors_mean)

        # 2nd moment of reconstructed tensor <[|..|]**2>
        self.posteriors.tensor_m2 = cp_to_tensor(get_AAt(factors_mean, factors_variance))

    def variational_inference(self, observed_tensor):
        """ Variational Expectation Maximization  """
        self._init_inference(observed_tensor)

        # Log every disppct batches.
        disppct = self.fit_params.disppct
        ite_max = self.fit_params.ite_max
        precn = np.floor(1 + np.log10(ite_max)).astype(int)

        self.loss_tot = []
        self.shape_param_tot = []

        for ite in range(ite_max):

            # Variational Expectation Update: Latent U
            self._update_latent(observed_tensor)

            # Variational Expectation Update: CP factors [|A1,..,AD|]
            if np.any(self.fit_params.fit_factors_dim):
                self._update_factors(observed_tensor)

            # Variational Expectation Update: Offset  V
            if np.any(self.fit_params.fit_offset_dim):
                self._update_offset(observed_tensor)

            # Variational Expectation Update: Precision for ARD
            if np.any(self.fit_params.shared_precision_dim):
                self._update_precision_shared_ard(observed_tensor)

            # Variational Expectation Update: Precision with mode (neuron) groups
            if not(self.fit_params.shared_precision_mode is None):
                self._update_precision_shared_mode(observed_tensor)

            # Variational Maximization Update: Shape parameter E
            if self.fit_params.fit_shape_param:
                self._update_shape_param(observed_tensor)

            # Print Loss
            if ite % disppct == 0:
                loss = self.loss_tot[-1]
                shape = self.shape_param_tot[-1]
                print("Iterations: %s / %d | loss =   %.4f | shape =   %.4f" %
                      (str(ite).zfill(precn), ite_max, loss, shape))

    def _update_shape_param(self, observed_tensor):
        """Variational Maximization Step"""
        # Update method
        shape_update = self.fit_params.shape_update

        # Dear with missing entries
        observed_data = self.fit_params.observed_data
        if not np.all(observed_data == 1):
            observed_id = np.where(observed_data)
        else:
            observed_id = np.where(np.ones(self.shape))

        if shape_update == 'MM-G':

            # Restrict Dataset for fast shape update
            #  batch_size = 25000
            batch_size = len(observed_id[0])
            keep = np.random.permutation(len(observed_id[0]))
            keep = keep[:batch_size]
            observed_id = tuple([xi[keep] for xi in observed_id])

            # Observed_data
            observed_tensor_reduced = np.expand_dims(observed_tensor[observed_id], axis=1)

            # 1st and 2nd moment of low rank tensor
            tensor_m1 = np.expand_dims(self.posteriors.tensor_m1[observed_id], axis=1)
            tensor_m2 = np.expand_dims(self.posteriors.tensor_m2[observed_id], axis=1)

            # 1st and 2nd moment of offset
            offset_m1 = self.posteriors.offset_mean
            offset_m2 = self.posteriors.offset_mean**2 + self.posteriors.offset_variance

            offset_m1 = np.expand_dims(offset_m1[observed_id], axis=1)
            offset_m2 = np.expand_dims(offset_m2[observed_id], axis=1)

            # PG params
            shape_old = self.shape_param
            omega = np.sqrt(tensor_m2 + offset_m2 + 2*tensor_m1*offset_m1)

            # Moment match PG(1, Om) with Gamma(alpha0,beta0)
            moments = pg_moment(np.ones(batch_size), omega, centered=1)
            alpha_0 = np.expand_dims(moments[:, 0]**2 / moments[:, 1], axis=1)

            def minus_free_energy(shape_test):
                shape_test = np.expand_dims(shape_test, axis=0)

                FE = np.sum(
                    + gammaln(observed_tensor_reduced + shape_test)
                    - gammaln(shape_test)
                    - gammaln(alpha_0*(observed_tensor_reduced + shape_test))
                    + shape_test * (
                            alpha_0 * psi(alpha_0*(observed_tensor_reduced + shape_old))
                            - np.log(2)
                            - 0.5 * (tensor_m1 + offset_m1)
                            - np.log(np.cosh(omega/2)))
                    , axis=0)
                return -FE

            # Maximize shape dependent free energy
            shape_new = scipy.optimize.fmin(func=minus_free_energy, x0=shape_old, disp=False)

            loss_old = minus_free_energy(shape_old)
            loss_new = minus_free_energy(shape_new)

            if loss_old < loss_new:
                print('Shape optimization likely failed, continue with previous value')
                shape_new = shape_old
                loss_new = loss_old

            # Constant (wrt. shape) part of Free Energy (ELBO)
            minus_free_energy_0 = np.sum(
                - observed_tensor_reduced*(0.5*(tensor_m1 + offset_m1) - np.log(np.cosh(0.5*omega)))
                + shape_old*alpha_0*psi(alpha_0*(observed_tensor_reduced+shape_old))
                - gammaln(alpha_0*(observed_tensor_reduced+shape_old)))

            minus_free_energy_kl = self._kl_factors() \
                                   + self._kl_offset() \
                                   + self._kl_precision_shared() \
                                   + self._kl_precision_mode()

            loss_final = loss_new[0] + minus_free_energy_0 + minus_free_energy_kl

            self.shape_param = shape_new
            self.loss_tot.append(loss_final)
            self.shape_param_tot.append(shape_new[0])

        else:
            raise NameError('Shape Update Not Implemented')

        return

    def _kl_factors(self):
        """KL Divergences involving Gaussian distributed rows of the factors"""

        KL = 0.0

        if any(self.fit_params.fit_factors_dim):

            factors_posterior_mean = self.posteriors.factors_mean
            factors_posterior_variance = self.posteriors.factors_variance
            factors_priors_precision = self.priors.factors_precision

            tensor_rank = self.rank

            for dim_ext in np.arange(len(self.shape)):
                for dim_int in np.arange(self.shape[dim_ext]):
                    m = factors_posterior_mean[dim_ext][dim_int, :]
                    v = np.reshape(factors_posterior_variance[dim_ext][dim_int, :], (tensor_rank, tensor_rank))
                    p = np.reshape(factors_priors_precision[dim_ext][dim_int, :], (tensor_rank, tensor_rank))

                    ds = p @ v

                    KL += 0.5 * np.trace(ds) \
                          - np.log(np.linalg.det(ds)) \
                          - np.squeeze(np.expand_dims(m, axis=0) @ p @ np.expand_dims(m, axis=1)) \
                          - tensor_rank

        return KL


    def _kl_offset(self):
        """KL Divergences involving Gaussian distributed constrained offset"""

        if self.fit_params.fit_offset:
            to_keep = np.where(self.fit_params.fit_offset_dim)[0]
            to_remo = np.where(1-np.array(self.fit_params.fit_offset_dim))[0]

            new_order = tuple(np.concatenate((to_keep, to_remo)))
            new_length = np.prod([self.shape[i] for i in to_keep])

            means = np.reshape(np.transpose(self.posteriors.offset_mean, new_order), (-1))[:new_length]
            varis = np.reshape(np.transpose(self.posteriors.offset_variance, new_order), (-1))[:new_length]
            preci = np.reshape(self.priors.offset_precision, (-1))

            KL = np.sum(0.5 * varis *preci - np.log(varis*preci) - means**2*preci -1)

        else:
            KL = 0

        return KL




    def _kl_precision_mode(self):
        """KL Divergences involving Gamma distributed precision priors"""
        if not(self.fit_params.shared_precision_mode is None):

            a_posterior = self.posteriors.a_mode
            b_posterior = self.posteriors.b_mode

            a_prior = self.priors.a_mode
            b_prior = self.priors.b_mode

            KL = np.sum(_kl_gamma(a_posterior, a_prior, b_posterior, b_prior))

        else:
            KL = 0

        return KL

    def _kl_precision_shared(self):
        """KL Divergences involving Gamma distributed precision priors"""
        if any(self.fit_params.shared_precision_dim):
            a_posterior = self.posteriors.a_shared
            b_posterior = self.posteriors.b_shared

            a_prior = self.priors.a_shared * np.ones(len(a_posterior))
            b_prior = self.priors.b_shared * np.ones(len(a_posterior))

            KL = np.sum(_kl_gamma(a_posterior, a_prior, b_posterior, b_prior))

        else:
            KL = 0

        return KL


def _kl_gamma(alpha1, alpha2, beta1, beta2):
    """" KL divergence of Gamma distributions """

    alpha1 = np.reshape(alpha1, (-1))
    alpha2 = np.reshape(alpha2, (-1))
    beta1 = np.reshape(beta1, (-1))
    beta2 = np.reshape(beta2, (-1))

    k1 = (alpha1 - alpha2) * psi(alpha1)
    k2 = gammaln(alpha2) - gammaln(alpha1)
    k3 = alpha2 * (np.log(beta1) - np.log(beta2))
    k4 = alpha1 * (beta2 - beta1) / beta1

    return k1 + k2 + k3 + k4

