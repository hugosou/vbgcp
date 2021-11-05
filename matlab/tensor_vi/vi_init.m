function [vi_var,vi_param] = vi_init(Xobs, vi_param, vi_var)
% Initialize Variational Inference Tensor Decomposition

if nargin <3
    vi_var = struct();
end

% Tensor Rank
R = vi_param.R;

% Dimension of the problem
Xdims = size(Xobs);

% Fit offset ?
fit_offset_dim = vi_param.fit_offset_dim;
fit_offset = sum(fit_offset_dim)>0;

%% Priors

% Priors CP: precision and mean
if not(isfield(vi_var,'CP_prior_mean'))
    vi_var.CP_prior_mean = ...
        cellfun(@(Z) zeros(Z,R) , num2cell(Xdims),'UniformOutput', false);        
end

if not(isfield(vi_var,'CP_prior_precision'))
    vi_var.CP_prior_precision =  ...
        cellfun(@(Z) 0.01*repmat(reshape(eye(R),[1,R*R]),Z,1) , num2cell(Xdims),'UniformOutput', false);
end

% Prior Offset: precision and mean
if not(isfield(vi_var,'offset_prior_mean'))
    if sum(fit_offset_dim)==1
        vi_var.offset_prior_mean = zeros(Xdims(find(fit_offset_dim)),1);
    else
        vi_var.offset_prior_mean = zeros(Xdims(find(fit_offset_dim)));
    end
end

if not(isfield(vi_var,'offset_prior_precision'))
    if sum(fit_offset_dim)==1
        vi_var.offset_prior_precision = 0.00001*ones(Xdims(find(fit_offset_dim)),1);
    else
        vi_var.offset_prior_precision = 0.00001*ones(Xdims(find(fit_offset_dim)));
    end
end


% Precision prior
if not(isfield(vi_var,'prior_a_mode'))
    prior_a_mode = 100; vi_var.prior_a_mode = prior_a_mode;
end

if not(isfield(vi_var,'prior_b_mode'))
    prior_b_mode = 1; vi_var.prior_b_mode = prior_b_mode;
end


% Precision prioe
if not(isfield(vi_var,'prior_a_shared'))
    prior_a_shared = 100; vi_var.prior_a_shared=prior_a_shared;
end

if not(isfield(vi_var,'prior_b_shared'))
    prior_b_shared = 1; vi_var.prior_b_shared=prior_b_shared;
end

%% Init variational distributions

% Init Offset
if all(vi_param.observed_data(:)==1)
    observed_data = 1;
    observed_id = (1:numel(Xobs))';
else
    observed_data= vi_param.observed_data;
    observed_id = find(observed_data);
end

if not(isfield(vi_var,'offset_mean'))
    vi_var.offset_mean = fit_offset*init_offsets(Xdims, fit_offset_dim);
    vi_var.offset_mean = observed_data.*vi_var.offset_mean;
end 

if not(isfield(vi_var,'offset_variance'))
    vi_var.offset_variance = fit_offset*abs(init_offsets(Xdims, fit_offset_dim));
    vi_var.offset_variance = observed_data.*vi_var.offset_variance;
end

% Init CP
if not(isfield(vi_var,'CP_mean'))
    vi_var.CP_mean = cellfun(@(Z) rand(Z,R) , num2cell(Xdims),'UniformOutput', false);
    %vi_var.CP_mean = cellfun(@(Z) 0.1*rand(Z,R) , num2cell(Xdims),'UniformOutput', false);
end

if not(isfield(vi_var,'CP_variance'))
    vi_var.CP_variance = init_CP_variance(Xdims,R);
end

%% Init Shape 

if not(isfield(vi_var,'shape'))
    vi_var.shape = mean(Xobs(observed_id(:)));
end

%% Init Latents
    
if not(isfield(vi_var,'latent_mean'))
    % Get reconstructed tensor 1st and second moment
    AAt = get_AAt(vi_var.CP_mean,vi_var.CP_variance);
    tensor_mean  = tensor_reconstruct(vi_var.CP_mean);
    tensor2_mean = tensor_reconstruct(AAt);
    
    % Save tensor moments
    vi_var.tensor_mean  = tensor_mean;
    vi_var.tensor2_mean = tensor2_mean;
    
    % Variational Update: Latent U
    vi_var = vi_update_latent(vi_var,vi_param,Xobs);
end

%% Store Observed data for efficient CP update
if isfield(vi_param,'sparse')
    if strcmp(vi_param.sparse,'block')
        vi_var = store_observed(vi_var,vi_param,Xobs);
    elseif strcmp(vi_param.sparse,'false')
        %
    else
        error('Sparsity mode not Implemented')
        
    end
else
    vi_param.sparse = 'false';
end



end

function CP_variance = init_CP_variance(Xdims,R)
    CP_variance  = cellfun(@(Z) ...
        repmat(reshape(eye(R),[1,R*R]),Z,1) , num2cell(Xdims),'UniformOutput', false);
    
    for dimn=1:length(Xdims)
        for dimi=1:size(CP_variance{1,dimn},1)
            tmp = 0.01*randn(R,R);
            tmp = 0.1*randn(R,R);
            tmp = tmp'*tmp;
            CP_variance{1,dimn}(dimi,:) = tmp(:);
        end
    end

    
end

function offsets = init_offsets(Xdims, fit_offset_dim)
% Init Offset

%offsets_tmp = 0.01*randn(fit_offset_dim.*Xdims+not(fit_offset_dim));
offsets_tmp = 0*rand(fit_offset_dim.*Xdims+not(fit_offset_dim));
%offsets_tmp = rand(fit_offset_dim.*Xdims+not(fit_offset_dim));
offsets     = repmat(offsets_tmp, fit_offset_dim + not(fit_offset_dim).*Xdims);
 
end

function vi_var = store_observed(vi_var,vi_param,Xobs)
% Store observed data for efficient sparse data CP update
% NOTE: Currently exploit block structure of the 'pre-ordered' dataset
%       More efficient implementations might depend on dataset.
%
% OUPUTS: - observed_data_block: cell array containing boolean of observed
%         data shared accross blocks
%           
%         - observed_data_bloc_id: cell array containg corresponding rows
%         for each mode


% Dimensions of the problem
Xdims = size(Xobs);

% Boolean of Observed entries 
observed_data = vi_param.observed_data;
if all(observed_data(:) ==1)
    observed_data =ones(Xdims);
end

% Gather Observed entries for each row of factor matrices
observed_data_ni = cell(1,length(Xdims));

for dimn = 1:length(Xdims)
    On = tensor_unfold(observed_data,dimn);
    observed_data_ni{dimn} = cell(1, Xdims(dimn));
    
    for dimi = 1:Xdims(dimn)
        observed_data_ni{dimn}{dimi} = find(On(dimi,:));
    end
end

% Try to find a block structure (can be improved)
observed_data_block = cell(1,length(observed_data_ni));
observed_data_bloc_id = cell(1,length(observed_data_ni));

for dimn = 1:length(observed_data_ni)
    On = observed_data_ni{dimn};
    
    if all(cellfun(@(Z) size(Z,2),  On) == size(On{1},2))
        % Each factor rows have the same number of observed entries
        
        % Identify Blocks (should be 'pre-ordered')
        Onmat = cell2mat(On');
        isdif = sum(abs(diff(Onmat,1)),2);
        block_id = find(isdif);
        
        if isempty(block_id)
            % No blocks
            observed_data_block{1,dimn}{1} = On{1};
            observed_data_bloc_id{1,dimn}{1} = 1:length(On);
            
        else
            % Grasp Block ids and sizes
            block_id = [1;block_id+1];
            block_id_ext = [block_id(2:end);size(On,2)+1];
            
            observed_data_block{1,dimn} = On(1,block_id);
            observed_data_bloc_id{1,dimn} = cellfun(...
                @(U,V) U:(V-1),...
                num2cell(block_id),...
                num2cell(block_id_ext), 'UniformOutput',false )';
        end
    else
        % No Block Structure. Simply use sparsity
        observed_data_block{1,dimn} = On;
        observed_data_bloc_id{1,dimn} = num2cell(1:length(On));
    end
    
end

vi_var.observed_data_block=observed_data_block;
vi_var.observed_data_bloc_id=observed_data_bloc_id;

end

























