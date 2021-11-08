import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

""" utils functions to manipulate tensors and fit decompositions
    tensor: a D-dimensional array (size d1 x .. x dD)
    factors: a list containing D matrices [|A1, ..., AD|] of size d1xR, .., dD x R (rank-R CP decomposition)  
"""


def vectorize(tensor):
    """Unfold from tensor to vector"""
    return np.reshape(tensor, (-1))


def unfold(tensor, mode):
    """mode-th tensor unfolding"""
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def fold(matrix, mode, shape):
    """Fold back tensor tensor"""
    shape_tmp = [shape[mode], *np.delete(shape, mode)]
    return np.moveaxis(np.reshape(matrix, shape_tmp, order='F'), 0, mode)


def zeros_factors(shape, rank):
    """CP decomposition of zeros"""
    return [np.zeros(shape=[dim, rank]) for dim in shape]


def eyes_precisions(shape, rank, weight=1):
    """Init precision diagonal matrices"""
    return [weight * np.tile(np.reshape(np.eye(rank), (-1)), (dim, 1)) for dim in shape]


def rand_factors(shape, rank, weight=1):
    """Random CP factors"""
    return [weight * np.random.rand(dim, rank) for dim in shape]


def plot_factors(factors, variances=None, color='k', nstd=1):
    """Plot a CP decomposition"""
    rank = check_rank_factors(factors)
    shape = get_dim_factors(factors)

    if not(variances is None):
        vrank = check_rank_factors(variances)
        vshape = get_dim_factors(variances)
        diag_id = np.arange(rank) * rank + np.arange(rank)
        assert vrank == rank**2
        assert vshape == shape
        plot_variance = True
    else:
        plot_variance = False

    for ii in range(rank):
        for jj in range(len(shape)):

            pcur = factors[jj][:, ii]
            xcur = np.arange(shape[jj])

            plt.subplot(rank, len(shape), 1 + jj + ii * len(shape))
            plt.scatter(xcur, pcur, c=color)

            if plot_variance:
                vcur = variances[jj][:, diag_id[ii]]
                up = pcur + nstd * np.sqrt(vcur)
                lo = pcur - nstd * np.sqrt(vcur)
                plt.fill_between(xcur, lo,  up, color=color, alpha=0.2)

            if ii == 0:
                plt.title('Dim.' + str(jj + 1))
            if jj == 0:
                plt.ylabel('Comp.' + str(ii + 1))


def get_rank_factors(factors):
    """Get the size fo the first factor"""
    return factors[0].shape[1]


def check_rank_factors(factors):
    """Check the rank of a factors list"""
    rank = factors[0].shape[1]
    assert all([factors[dim].shape[1] == rank for dim in range(len(factors))]), "Incorrect factors dimension"
    return rank


def get_dim_factors(factors):
    """Get the reconstructed tensor dimension"""
    return [factors[dim].shape[0] for dim in range(len(factors))]


def khatri_rao_AB(A, B, rank):
    """ Khatri-Rao Product of two matrices A and B with same rank """

    a = A.shape[0]
    b = B.shape[0]
    kr = np.reshape(A, [a, 1, rank]) * np.reshape(B, [1, b, rank])
    return np.reshape(kr, (a*b, rank))


def khatri_rao(matrices, reverse=False, skip=[]):
    """ Khatri-Rao Product of a list of matrices
            reverse: the order of KR product
            skip: some matrices in the list
    """

    # Check that matrices have correct sizes
    rank = check_rank_factors(matrices)

    # IDs to iterate over element in list
    all_dim = [i for i in range(len(matrices)) if not(i in skip)]

    # Reverse iteration order
    if reverse:
        all_dim.reverse()

    # First Matrix
    kr = matrices[all_dim[0]]

    # Iterates the KR product
    for dim in all_dim[1:]:
        kr = khatri_rao_AB(kr, matrices[dim], rank)

    return kr


def cp_to_tensor(factors):
    """ Reconstruct a tensor from its CP decomposition """

    if len(factors) > 1:
        dims = get_dim_factors(factors)

        # Mode(0) using MTTKRP representation
        tensor = factors[0] @ np.transpose(khatri_rao(factors, reverse=True, skip=[0]))

        # Fold Back
        tensor = fold(tensor, 0, dims)

    else:
        tensor = factors[0]

    return tensor


def normalize_cp(factors, dosort=1, normdim=None):
    """Normalise a CP decomposition and inject the normaliser in normdim if provider"""
    # TODO implement the normalization of the variance ?

    normed_factors = [factori for factori in factors]
    norms = 1

    for dim, U in enumerate(normed_factors):

        # Get norm of dim-th factor
        normi = np.sqrt(np.diag(np.transpose(U) @ U))

        # Normalize dim-th factor
        U = (U / (normi + 1e-12))

        # Convention for CP factors sign
        signor = np.sign(np.sum(U, axis=0))
        signor[np.nonzero(signor == 0)] = 1

        U *= signor
        norms *= signor

        # Reassign factor
        normed_factors[dim] = U

        np.where(signor == 1)
        # Gather norms
        norms *= normi

    # Norms are defined positive: inject sign in last dim if not.
    normed_factors[-1] *= np.sign(norms)
    norms *= np.sign(norms)

    # Sort the factors
    if dosort:
        sort_id = np.argsort(norms)
        sort_id = sort_id[::-1]

        normed_factors = [facti[:,sort_id] for facti in normed_factors]
        norms = norms[sort_id]

    # Inject back norms in a given dimension
    if not(normdim is None):
        normed_factors[normdim] *= norms
        norms = np.ones(len(factors))

    return normed_factors, norms


def get_similarity(models, ref_model=0, used_dims=None):
    """ Estimate similarity index from list of CP-Tensor factors models
        Factors must have the same rank
        Metric from (Tomasi & Bros 2004 and Williams et al (2018)) adapted with Munkres (Hungarian) Algorithm.
        Output are similarity metrics, permutation and sign shift necessary to align normed factors"""

    # Dimensions of the problem
    tensor_rank = check_rank_factors(models[ref_model])
    tensor_shape = get_dim_factors(models[ref_model])
    tensor_dim = len(tensor_shape)
    num_model = len(models)

    # Dimensions used to calculate similarities
    if used_dims is None:
        used_dims = np.ones(len(tensor_shape), dtype=bool)

    # Normalize reference model
    ref_model, ref_norms = normalize_cp(models[ref_model], dosort=1, normdim=None)

    # Init
    smlty = []
    sign_final = []
    perm_final = []

    for (i, cur_model) in enumerate(models):

        # Normalize current model
        cur_model, cur_norms = normalize_cp(cur_model, dosort=1, normdim=None)

        # Calculate a_r a_p(r) o b_r b_p(r) o ... for all permutations
        cur_product = np.ones((tensor_rank, tensor_rank))
        signtot = []
        for dim_ext in np.arange(tensor_dim):
            vref = ref_model[dim_ext]
            vcur = cur_model[dim_ext]
            signtot.append(np.transpose(np.sign(np.transpose(vref) @ vcur)))

            if used_dims[dim_ext]:
                cur_product *= np.transpose(vref) @ vcur
            else:
                cur_product *= np.sign(np.transpose(vref) @ vcur)

        # Take factors norms into account
        dif12 = np.abs(np.expand_dims(cur_norms, axis=0) - np.expand_dims(ref_norms, axis=1))
        max12 = 0.5 * np.abs(np.abs(np.expand_dims(cur_norms, axis=0) + np.expand_dims(ref_norms, axis=1))) \
                + 0.5 * np.abs(np.abs(np.expand_dims(cur_norms, axis=0) - np.expand_dims(ref_norms, axis=1)))

        # Get the full similarities for ALL permutations
        rapp = np.sum(ref_norms > 1e-15)  # In case some CP are zeros
        cur_product = (1 - dif12 / (max12 + 1e-15)) * cur_product / rapp

        # Use Munkres (Hungarian) Algorithm for Linear Assignment Problem on cur_product
        cur_col, cur_perm = linear_sum_assignment(-cur_product)

        # Similarities
        cur_sum = np.sum(cur_product[cur_col, cur_perm])

        # Sign shift necessary to align factors
        cur_sig = [signi[cur_perm, cur_col] for signi in signtot]
        cur_sig = np.reshape(np.concatenate(cur_sig), (tensor_rank, tensor_dim), order='F')
        cur_sig[cur_sig == 0] = 1

        # Store similarity index and corresponding permutation
        smlty.append(cur_sum)
        sign_final.append(cur_sig)
        perm_final.append(cur_perm)
        # smlty(find(isnan(smlty))) = 0;

    return smlty, perm_final, sign_final


def reorder_models(models, ref_model=0, permutations=None, sign_shift=None):
    """Align CP models based on a similarity metric"""

    if (permutations is None) or (sign_shift is None):
        print('No ordering provided. Ordering with default values')
        smlty, permutations, sign_shift = get_similarity(models, ref_model=ref_model)

    num_models = len(permutations)
    assert num_models == len(permutations)
    assert num_models == len(sign_shift)

    models_ordered = []

    for (i, cur_model) in enumerate(models):

        cur_model, _ = normalize_cp(cur_model, dosort=1, normdim=3)
        cur_perm = permutations[i]
        cur_sign = sign_shift[i]

        new_model = []

        for dim_ext in np.arange(len(cur_model)):
            new_model.append(cur_model[dim_ext][:, cur_perm] * np.expand_dims(cur_sign[:, dim_ext], axis=0))

        models_ordered.append(new_model)

    return models_ordered


def expand_factors(factors, r_new):
    """Add or remove CP component
       For the latter, we keep the ones with biggest amplitude
    """
    r_cur = get_rank_factors(factors)
    tensor_dim = get_dim_factors(factors)
    factors, _ = normalize_cp(factors, normdim=3)

    if r_new > r_cur:
        r_add = r_new-r_cur
        factors_new = [np.concatenate((fact, np.zeros((dim, r_add))), axis=1) for (fact, dim) in zip(factors, tensor_dim)]
    elif r_new < r_cur:
        factors_new = [fact[:, :r_new] for fact in factors]
    else:
        factors_new = factors

    return factors_new


def get_MMt(factors):
    """
    For factors = [|A1, ..., AD|]
    Returns [|AA1, ..., AAD|] where AAi[j,:] = Ai[j,:]' Ai[j,:] ~ 1 x rank^2
    """
    return [khatri_rao_AB(Ai.transpose(), Ai.transpose(), Ai.shape[0]).transpose() for Ai in factors]


def get_AAt(factors_mean, factors_variance):
    """
    For factors = [|A1, ..., AD|]
    Returns [|AA1, ..., AAD|] where AAi[j,:] = <Ai[j,:]' Ai[j,:]> ~ 1 x rank^2
    """
    MMt = get_MMt(factors_mean)
    return [mmt + v for (mmt, v) in zip(MMt, factors_variance)]


def compact_to_full_offset(compact_tensor, full_shape, fit_offset_dim):
    """Reconstruct tensor of size full_shape
    from the compact representation of a tensor allowed to vary across fit_offset_dim"""
    assert len(full_shape) == len(fit_offset_dim)

    # Shape of the compact form
    offset_shape = [full_shape[ii] for ii in np.where(fit_offset_dim)[0]]

    # Dimensions along which offset can vary
    to_ones = np.where(fit_offset_dim)[0]

    # Dimensions along which offset is tiled
    to_repeat = np.where(1 - np.array(fit_offset_dim))[0]

    # For reordering
    tmp_permute = np.concatenate((to_repeat[:], to_ones[:]))
    inv_permute = np.arange(tmp_permute.size)
    for i in np.arange(tmp_permute.size):
        inv_permute[tmp_permute[i]] = i

    # Tile offset
    full_tensor = np.tile(compact_tensor, np.concatenate(([full_shape[i] for i in to_repeat], [1 for _ in to_ones])))

    # Reshape
    return np.transpose(full_tensor, inv_permute)


def pg_smoother(c, l, k):
    """Smooth the limit of Polya-Gamma Laplace derivatives to avoid numerical divergences"""
    return np.exp(-1/(c/l)**k)


def pg_lap1(c):
    """PG(1,c) Laplace 1st derivatives"""
    return -1*(1/(2*c))*np.tanh(c/2)


def pg_lap2_tmp(c):
    """PG(1,c) Laplace 2nd derivatives"""
    return (1/(4*(np.cosh(c/2)**2)*c**3))*(np.sinh(c)-c)


def pg_lap3_tmp(c):
    """PG(1,c) Laplace 3rd derivatives"""
    return (1/(4*(np.cosh(c/2)**2)*c**5))*(c**2.*np.tanh(c/2) + 3*(c-np.sinh(c)))


def pg_lap2(c):
    """Continuous PG(1,c) Laplace 2nd derivatives"""
    k = 2
    l = 0.01
    return pg_lap2_tmp(c)*pg_smoother(c, l, k) +1/24.*(1-pg_smoother(c, l, k))


def pg_lap3(c):
    """Continuous PG(1,c) Laplace 3rd derivatives"""
    k = 2
    l = 0.01
    return pg_lap3_tmp(c)*pg_smoother(c, l, k) -1/60.*(1-pg_smoother(c, l, k))


def pg_moment(b, c, centered=1):
    """ First moments of Polya-Gamma distribution
        For phi log laplace phi = log < exp(-ut)>
        Derive and apply at 0"""

    b = np.reshape(b, (-1))
    c = np.reshape(c, (-1)) + 1e-16

    phi_1 = b*pg_lap1(c)
    phi_2 = b*pg_lap2(c)
    phi_3 = b*pg_lap3(c)

    if centered:
        moments = np.array([-phi_1, phi_2, -phi_3])
    else:
        m1 = -phi_1
        m2 = phi_2 + phi_1**2
        m3 = 2 * phi_1**3 - phi_3 - 3 * phi_1 * (phi_2 + phi_1 ** 2)
        moments = np.array([m1, m2, m3])

    return np.transpose(moments)


