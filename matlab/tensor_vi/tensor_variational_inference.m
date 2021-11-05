function vi_var = tensor_variational_inference(Xobs,vi_param,vi_var)
% Bayesian tensor CP decomposition of count data 
% Inference via approx. Variational Inference and Polya-Gamma  augmentation. 
%
% PARAMS: - Xobs: count tensor
%
%         - vi_param: structure containing fitting parameters 
%           - R: tensor rank
%           - ite_max: variational EM iterations
%           - (optional) fit_offset_dim: boolean, dimension allong which
%           offset varies
%           - (optional) observed_data: boolean, missing/observed entries
%           - (optional) dim_neuron: int, shared precision for a given mode
%           - (optional) neurons_groups: boolean, groups for shared
%           precision
%           - (optional) shared_precision_dim: boolean, shared precision
%           across given modes
%           - (optional) sparse: block structure for missing data
%           - (optional) disppct: display loss
%
%         - vi_var: structure containing model params. priors and posteriors
%           - CP mean (cell 1xD containing Di x R matrices): prior and post.
%           - CP precision/variance (cell 1xD containing Di x (RxR) matrices): prior and post.
%           - tensor_mean 1st  moment of reconstructed tensor from CP_mean
%           - tensor2_mean 2nd moment of reconstructed tensor from CP_mean
%           - Offset mean/variance priors and posteriors
%           - Gamma priors for precisions (prior_a, prior_b shared or mode)
%           - FE approximate free energy (ELBO)
%
% See Soulat et al. (2021)

if not(isfield(vi_param,'ite_max')); error('ite_max required field'); else; ite_max = vi_param.ite_max; end
if not(isfield(vi_param,'shape_update')); vi_param.shape_update = 'MM-G'; end
if not(isfield(vi_param,'disppct')); vi_param.disppct = 1; end

% Optional : Automatic Relevance Determination Parameter
if not(isfield(vi_param,'dim_neuron')); vi_param.dim_neuron = 0; end
if not(isfield(vi_param,'shared_precision_dim')); vi_param.shared_precision_dim = 0; end

% Optional : Missing Data
if not(isfield(vi_param,'observed_data'));   vi_param.observed_data = 1; end

% Optional : Dimensions to fit
if not(isfield(vi_param,'update_CP_dim')); vi_param.update_CP_dim  = ones(1,ndims(Xobs)); end
if not(isfield(vi_param,'fit_offset_dim'));vi_param.fit_offset_dim = ones(1, ndims(Xobs)); end

% Init variational variables
if nargin <3; vi_var = struc(); end
[vi_var,vi_param] = vi_init(Xobs, vi_param,vi_var);

loss_tot  = zeros(ite_max,1);
shape_tot = zeros(ite_max,1);

ref = vi_var.CP_mean;
precn = max(1+floor(log10(ite_max)),1);
logger = ['Iterations: %.', num2str(precn),'d/%d %s%.10g %s%.5g \n'];

for ite=1:ite_max
    
    % Variational Update: Latent U
    vi_var = vi_update_latent(vi_var,vi_param,Xobs);
    
    % Variational Update: CP factors
    vi_var = vi_update_CP(vi_var,vi_param,Xobs);
        
    % Variational Update: Offset V
    if sum(vi_param.fit_offset_dim)>0
        vi_var = vi_update_offset(vi_var,vi_param,Xobs);
    end
    
    % Variational Update: Precision for ARD
    if sum(vi_param.shared_precision_dim)>0
        vi_var = vi_update_precision_shared(vi_var,vi_param);
    end
    
    % Variational Update: Precision with Neuron Groups
    if vi_param.dim_neuron>0
        vi_var = vi_update_precision_mode(vi_var,vi_param);
    end

    % Variational Update: Shape
    vi_var = vi_update_shape(vi_var,vi_param, Xobs);
    
    % Display Loss 
    [loss,loss_str,ref] = get_loss(vi_var,vi_param,ref);
    display_loss = mod(100 * ite/ite_max,vi_param.disppct)==0 && vi_param.disppct>0;
    if display_loss
        fprintf(logger,ite,ite_max,loss_str,loss,'| Shape = ',vi_var.shape); 
    end
    
    loss_tot(ite)  = loss;
    shape_tot(ite) = vi_var.shape;
    
end

vi_var.loss_tot  = loss_tot; 
vi_var.shape_tot = shape_tot;

end

function [loss,loss_str,ref] = get_loss(vi_var,vi_param,ref)
if contains(vi_param.shape_update,'MM') || strcmp(vi_param.shape_update,'numerical')
    loss_str = '| FE = ';
    loss = vi_var.FE;
else
    loss_str = '| dSim = ';
    loss = get_similarity({ref, vi_var.CP_mean}, 1); ref = vi_var.CP_mean;
    loss = loss(2);
end
end



