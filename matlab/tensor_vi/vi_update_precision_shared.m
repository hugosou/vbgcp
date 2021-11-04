function vi_var = vi_update_precision_shared(vi_var,vi_param)

shared_dim = vi_param.shared_precision_dim;

% Grasp Current Posteriors
CP_mean     = vi_var.CP_mean;
CP_variance = vi_var.CP_variance;

% Grasp previous precisions
prior_precision_old = vi_var.CP_prior_precision;

% Tensor Ransk
R = size(CP_mean{1}, 2);

% Dimension of the problem
Xdims = cellfun(@(Z) size(Z,1), CP_mean);

% Prior mean 
prior_mean = vi_var.CP_prior_mean;

% Precision prior
prior_a_shared = vi_var.prior_a_shared;
prior_b_shared = vi_var.prior_b_shared;

% Each column (n) of DCP contains the Rx1 vector: diag (An'An)
DCP1 = cellfun(@(A,B) diag((A-B)'*(A-B)),CP_mean,prior_mean,'UniformOutput', false);
DCP1 = cell2mat(DCP1);

% Diagonal Element of Variance
diagid = find(eye(R))';
DCP2 = cellfun(@(Z) sum(Z(:,diagid),1)',CP_variance,'UniformOutput', false);
DCP2 = cell2mat(DCP2);

% <(a-mu)^2>
DCP = DCP1 + DCP2;

% Posterior Gamma parameters for shared precision diagonal
post_a = repmat(prior_a_shared + 0.5*sum(Xdims.*shared_dim), R,1);
post_b = prior_b_shared + 0.5*sum(DCP(:,find(shared_dim)),2);

% Updated Precision variational mean
prior_precision_mean = diag(post_a./post_b);

% Update relevant dimmensions of precision matrix
prior_precision_new = cellfun(@(X,Y,Z) (1-Y)*X + Y*repmat(prior_precision_mean(:)',Z,1),...
    prior_precision_old, num2cell(shared_dim),num2cell(Xdims),'UniformOutput', false);

% Update estimates
vi_var.CP_prior_precision = prior_precision_new;
vi_var.a_shared = post_a;
vi_var.b_shared = post_b;

end

