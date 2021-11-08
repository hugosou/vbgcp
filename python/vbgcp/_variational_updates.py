import numpy as np
from vbgcp.utils import get_AAt, unfold, khatri_rao, fold, compact_to_full_offset


class Mixin:
    """Variational Expectation Steps / Updates"""

    def _update_factors(self, observed_tensor):
        # TODO: implement block-sparse update

        # Grasp Variational Distributions
        latent = self.posteriors.latent_mean
        offset = self.posteriors.offset_mean
        shape_param = self.shape_param

        # Current Factors Posteriors
        factors_posterior_mean = self.posteriors.factors_mean
        factors_posterior_variance = self.posteriors.factors_variance

        # Current Factors Priors
        factors_prior_mean = self.priors.factors_mean
        factors_prior_precision = self.priors.factors_precision

        # Dimensions to be updated
        fit_factors_dim = self.fit_params.fit_factors_dim

        # Dimensions of the problem
        tensor_rank = self.rank
        tensor_shape = self.shape

        # Diagonal Elements for Precision
        diag_id = np.arange(tensor_rank)*(tensor_rank+1)

        # Deal with missing data
        observed_data = self.fit_params.observed_data

        for dim_ext in np.arange(len(tensor_shape)):
            if fit_factors_dim[dim_ext]:
                # Loop on dimensions: n-th Unfoldings

                Z = ((observed_tensor-shape_param)/2 - offset * latent) * observed_data
                Zn = unfold(Z, dim_ext)
                Un = unfold(latent, dim_ext)
                Bn = khatri_rao(factors_posterior_mean,
                                reverse=True, skip=[dim_ext])
                BBn = khatri_rao(get_AAt(factors_posterior_mean, factors_posterior_variance),
                                 reverse=True, skip=[dim_ext])

                # Priors
                prior_mean = factors_prior_mean[dim_ext]
                prior_precision = factors_prior_precision[dim_ext]

                # <B'UB>
                BUB = Un @ BBn

                # <B'><U><Z>
                BUZ = Zn @ Bn

                # Temporary Update (precisions are diag)
                posterior_precision = prior_precision + BUB
                posterior_mean_tmp = BUZ + prior_precision[:, diag_id] * prior_mean

                # Invert Precision and Update mean
                for dim_int in np.arange(tensor_shape[dim_ext]):
                    # TODO: faster // loop on GPU
                    posterior_variance_i = np.linalg.inv(np.reshape(posterior_precision[dim_int, :],
                                                                    (tensor_rank, tensor_rank)))
                    posterior_mean_i = posterior_variance_i @ posterior_mean_tmp[dim_int, :]
                    posterior_variance_i = np.reshape(posterior_variance_i, (-1))

                    factors_posterior_mean[dim_ext][dim_int, :] = posterior_mean_i
                    factors_posterior_variance[dim_ext][dim_int, :] = posterior_variance_i

        # Use last unfolding to get <tensor> and <tensor^2>
        # TODO add tests for this !
        tensor_m1 = fold(factors_posterior_mean[dim_ext] @ np.transpose(Bn), dim_ext, tensor_shape)
        tensor_m2_tmp = np.kron(np.ones((1, tensor_rank)), factors_posterior_mean[dim_ext]) * \
            np.repeat(factors_posterior_mean[dim_ext], tensor_rank, axis=1)
        tensor_m2 = fold((tensor_m2_tmp + factors_posterior_variance[dim_ext]) @ np.transpose(BBn),
                         dim_ext, tensor_shape)

        # Save Posteriors
        self.posteriors.factors_mean = factors_posterior_mean
        self.posteriors.factors_variance = factors_posterior_variance

        # Save tensor moments
        self.posteriors.tensor_m1 = tensor_m1
        self.posteriors.tensor_m2 = tensor_m2

    def _update_offset(self, observed_tensor):
        """Variational update of the offset tensor constrained along fit_offset_dim"""

        # Current estimates and posteriors
        tensor = self.posteriors.tensor_m1
        latent = self.posteriors.latent_mean
        shape_param = self.shape_param

        # Priors
        offset_prior_mean = self.priors.offset_mean
        offset_prior_precision = self.priors.offset_precision

        # Problem dimensions
        tensor_shape = self.shape

        # Constrained Dimensions
        fit_offset_dim = self.fit_params.fit_offset_dim
        observed_data = self.fit_params.observed_data
        not_fit_offset_dim = np.where(1 - np.array(fit_offset_dim, dtype=bool))[0]

        # Auxiliary variables used in the augmented Gaussian Likelihood
        Ztmp = (observed_tensor - shape_param)/2 - tensor * latent
        Ztmp = Ztmp * observed_data

        Usum = np.sum(latent, axis=tuple(not_fit_offset_dim))
        Zsum = np.sum(Ztmp, axis=tuple(not_fit_offset_dim)) / (Usum + 1e-16)

        # Update offset mean and variance using prior mean and precision (compact form)
        offset_posterior_variance = 1 / (Usum + offset_prior_precision + 1e-16)
        offset_posterior_mean = offset_posterior_variance * (Usum * Zsum + offset_prior_precision * offset_prior_mean)

        # Tile and reshape
        offset_posterior_variance = compact_to_full_offset(offset_posterior_variance, tensor_shape, fit_offset_dim)
        offset_posterior_mean = compact_to_full_offset(offset_posterior_mean, tensor_shape, fit_offset_dim)

        # Update posteriors
        self.posteriors.offset_mean = offset_posterior_mean
        self.posteriors.offset_variance = offset_posterior_variance

        return

    def _update_latent(self, observed_tensor):
        # TODO: if sparse observation, more efficient implementation with only observed data

        # Use first and second moments to estimate PG(shape + observed_tensor, omega)
        omega = np.sqrt(
            self.posteriors.tensor_m2 + self.posteriors.offset_variance + self.posteriors.offset_mean ** 2 + 2 * self.posteriors.tensor_m1 * self.posteriors.offset_mean)

        # Posterior Mean
        latent_mean = ((self.shape_param + observed_tensor) / (2 * omega)) * np.tanh(omega / 2)
        self.posteriors.latent_mean = latent_mean * self.fit_params.observed_data

    def _update_precision_shared_mode(self, observed_tensor):
        # TODO implement for multiple modes

        mode = self.fit_params.shared_precision_mode
        groups = self.fit_params.neuron_groups

        # Grasp Current Factors Posteriors
        posterior_factor_mean = self.posteriors.factors_mean[mode]
        posterior_factor_variance = self.posteriors.factors_variance[mode]

        # Current Factors Priors
        prior_factor_mean = self.priors.factors_mean[mode]
        prior_factor_precision = self.priors.factors_precision[mode]

        # Mode Precision Parameters priors
        prior_a_mode = self.priors.a_mode
        prior_b_mode = self.priors.b_mode

        # Dimensions of the problem
        tensor_shape = self.shape
        tensor_rank = self.rank

        # Diagonal Element of Variance
        diag_id = np.arange(tensor_rank) * (tensor_rank + 1)

        # <(a-mu)^2> per group and component number: size R x num_group
        dCP2 = (posterior_factor_mean - prior_factor_mean)**2 + posterior_factor_variance[:, diag_id]
        dCP2 = [np.sum(dCP2 * np.expand_dims(groups[:, i], axis=1), axis=0) for i in np.arange(groups.shape[1])]
        dCP2 = np.reshape(np.concatenate(dCP2), (tensor_rank, groups.shape[1]), order='F')

        # Posterior Params: size R x num_group
        posterior_a_mode = np.tile(prior_a_mode + 0.5*np.sum(groups, axis=0), (tensor_rank, 1))
        posterior_b_mode = prior_b_mode + 0.5*dCP2

        # Precision
        mode_precision_tmp = posterior_a_mode/posterior_b_mode

        # Reorder in Ng x (RxR) precision matrix
        mode_precision = np.zeros((groups.shape[1], tensor_rank ** 2))
        mode_precision[:, diag_id] = np.transpose(mode_precision_tmp)

        # Assign each 'neuron' to the precision matrix of its group
        prior_factor_precision = mode_precision[np.argmax(groups, axis=1), :]

        # Update estimates
        self.priors.factors_precision[mode] = prior_factor_precision
        self.posteriors.a_mode = posterior_a_mode
        self.posteriors.b_mode = posterior_b_mode

        return NotImplementedError

    def _update_precision_shared_ard(self, observed_tensor):

        shared_precision_dim = self.fit_params.shared_precision_dim

        # Grasp Current Factors Posteriors
        posterior_factors_mean = self.posteriors.factors_mean
        posterior_factors_variance = self.posteriors.factors_variance

        # Current Factors Priors
        prior_factors_mean = self.priors.factors_mean
        prior_factors_precision = self.priors.factors_precision

        # Precision Prior Params
        prior_a_shared = self.priors.a_shared
        prior_b_shared = self.priors.b_shared

        # Each column (n) of DCP contains the Rx1 vector: diag (An'An)

        # Dimensions of the problem
        tensor_shape = self.shape
        tensor_rank = self.rank

        # Each column (n) of DCP contains the Rx1 vector: diag (An'An)
        centered_factors = [post - prio for post, prio in zip(posterior_factors_mean, prior_factors_mean)]
        DCP1 = [np.diag(np.transpose(cf) @ cf) for cf in centered_factors]
        DCP1 = np.reshape(np.concatenate(DCP1), (tensor_rank, len(tensor_shape)), order='F')

        # Diagonal Element of Variance
        diag_id = np.arange(tensor_rank) * (tensor_rank + 1)
        DCP2 = np.concatenate([np.sum(ivar[:, diag_id], axis=0) for ivar in posterior_factors_variance])
        DCP2 = np.reshape(DCP2, (tensor_rank, len(tensor_shape)), order='F')

        # <(a-mu)^2>
        DCP = DCP1 + DCP2

        # Posterior Gamma parameters for shared precision diagonal
        posterior_a_shared = np.repeat(
            prior_a_shared + 0.5 * np.sum(np.array(tensor_shape) * np.array(shared_precision_dim)),
            tensor_rank)
        posterior_b_shared = prior_b_shared + 0.5 * np.sum(DCP[:, np.where(shared_precision_dim)[0]], axis=1)

        # Updated Shared Precision variational mean
        prior_factors_precision_new = np.diag(posterior_a_shared / posterior_b_shared)

        # Update relevant dimensions of precision matrix
        prior_factors_precision_new = \
            [np.tile(np.reshape(prior_factors_precision_new, (-1)), (dim_int, 1)) * do_update + (1-do_update) * old
             for dim_int, old, do_update in zip(tensor_shape, prior_factors_precision, shared_precision_dim)]

        # Update estimates
        self.priors.factors_precision = prior_factors_precision_new
        self.posteriors.a_shared = posterior_a_shared
        self.posteriors.b_shared = posterior_b_shared

        return