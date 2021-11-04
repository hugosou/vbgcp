function vi_var = vi_update_precision_mode(vi_var,vi_param)

dimn   = vi_param.dim_neuron;
groups = vi_param.neurons_groups;

% Grasp Current Posteriors
CP_mean     = vi_var.CP_mean;
CP_variance = vi_var.CP_variance;

% Prior Parameters
prior_mean = vi_var.CP_prior_mean;

% Precision prior
prior_a_mode = vi_var.prior_a_mode;
prior_b_mode = vi_var.prior_b_mode;

% Tensor Rank
R = size(CP_mean{1}, 2);

% Diagonal Element of Variance
diagid = find(eye(R))';

% <(a-mu)^2> per group and component number: size R x Ng
dCP2 = (CP_mean{dimn}-prior_mean{dimn}).^2 +  CP_variance{dimn}(:,diagid);
dCP2 = cell2mat(cellfun(@(X)  sum(X.*dCP2,1)' ,num2cell(groups,1),'UniformOutput', false));

% Posterior Params: size R x Ng
post_a = repmat(prior_a_mode + 0.5*sum(groups),R,1);
post_b = prior_b_mode + 0.5*dCP2;

% Varitional Mean Precision
posterior_precisions = post_a./post_b;

% Reorder in Ng x (RxR) precision matrix
posterior_precisions = cellfun(@(Z) diag(Z) , num2cell(posterior_precisions,1),'UniformOutput', false);
posterior_precisions = cell2mat(cellfun(@(Z) Z(:) , posterior_precisions,'UniformOutput', false))';

% Assign each 'neuron' to the precision matrix of its group
[~,neurons_groups_id] = max(groups, [],2);
prior_precision_new_dimn = posterior_precisions(neurons_groups_id,:);

% Update current precision
vi_var.CP_prior_precision{dimn} = prior_precision_new_dimn;

% Update current posterior
vi_var.a_mode = post_a;
vi_var.b_mode = post_b;

end