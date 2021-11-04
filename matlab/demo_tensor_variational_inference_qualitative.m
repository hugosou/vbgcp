%% Probabilistic Tensor Decomposition of Count Data
% Illustrate ARD for tensor rank (1/2): Missing Data + Offset
addpath(genpath('./'))

%% Generate Dataset
% Model
add_offset  = 1;
add_missing = 1;
model_true  = 'negative_binomial';

% Observed Tensor Dimensions
Xdims = [100,70,3,4,5];

% True Rank
Rtrue = 4;

% For Reproduction purposes
rng(1)

% Simulate Toy Dataset
[Xobs,observed_data,true_params] = ...
    build_toydataset(model_true,Rtrue,Xdims,add_offset,add_missing);

% Grasp simulated parameters
param_names = fieldnames(true_params);
for parami = 1:length(param_names)
    eval([param_names{parami} '=true_params.' param_names{parami},';']);
end

% Plot True dataset 
plot_cp(true_params.CPtrue)

% Variational Inference
clc
R = 6;

%%

% Fit parameters
vi_param = struct();
vi_param.ite_max = 1;
vi_param.observed_data = observed_data;
vi_param.fit_offset_dim = add_offset*fit_offset_dim;
vi_param.shared_precision_dim= 1*[0,1,1,1,1];
vi_param.dim_neuron= 1;
vi_param.neurons_groups = neurons_groups;
vi_param.update_CP_dim = ones(1,ndims(Xobs));
vi_param.shape_update = 'MM-G';
vi_param.R = R;

if add_missing
    vi_param.sparse = 'block';
else
    vi_param.sparse = 'false';
end

% Shared initialization
vi_var0 = struct();
vi_var0.shape = 80;
vi_param.disppct = 0.1;

% With ARD
[vi_var0,vi_param0] = vi_init(Xobs, vi_param, vi_var0);
vi_var_with_ard = tensor_variational_inference(Xobs,vi_param,vi_var0);

%%

% Without ARD
vi_param_no_ard = vi_param;
vi_param_no_ard.shared_precision_dim= 0*[0,1,1,1,1];
vi_param_no_ard.dim_neuron= 0;
vi_var_no_ard = tensor_variational_inference(Xobs,vi_param_no_ard,vi_var0);


%% Plots

% Choose method
vi_var= vi_var_with_ard;

% Fit summary
figure
subplot(1,2,1); hold on
plot(1:vi_param.ite_max, shape*ones(vi_param.ite_max,1), 'color','m', 'linewidth',2,'linestyle','--')
plot(1:vi_param.ite_max, vi_var_with_ard.shape_tot, 'color','k', 'linewidth',2)
box on; xlabel('Iteration'); title('Shape Parameter')
%ylim([0, 200])

subplot(1,2,2)
plot(1:vi_param.ite_max, vi_var_with_ard.loss_tot, 'color','k', 'linewidth',2)
box on; xlabel('Iteration'); title('Approximate FE')
set(gcf,'position',[1921         340         635         219])

% Compare fit and simulation
if R > Rtrue
    CPt = augment_cp(CPtrue,R-Rtrue);
else
    CPt = CPtrue;
end

if all(observed_data(:)==1)
    
    models = {CPt, vi_var.CP_mean};
    variances = {[],vi_var.CP_variance};
    
else
    
    % Gather neurons recorded only in some trials
    [mean,variance] = squeeze_missing(...
        vi_var.CP_mean,...
        vi_var.CP_variance,observed_data, [1,4]);
    
    [mean_true,~] = squeeze_missing(...
        CPt,...
        {},observed_data, [1,4]);
    
    
    models = {mean_true, mean};
    variances = {[],variance};
    
end

% Estimate Similarity Fit/True
[smlty_tot_noref,~,permt_tot,~,sig_tot] = ...
    get_similarity_tot(models,ones(1,ndims(Xobs)),1);



% Use it to reorder factors
[models,variances] = ...
    reorder_cps(models,permt_tot,sig_tot,variances);

% Plot
plot_cp(models{2},variances{2})
for dimn=1:size(models{1},2)
    for rr = 1:size(models{1}{1},2)
        subplot(size(models{1}{1},2), size(models{1},2), (rr-1)*size(models{1},2) + dimn)
        
        if dimn <3
            scatter(1:length(models{1}{1,dimn}(:,rr)), models{1}{1,dimn}(:,rr), 10,'m', 'filled')
        else
            scatter(1:length(models{1}{1,dimn}(:,rr)), models{1}{1,dimn}(:,rr), 40,'m', 'filled')
            scatter(1:length(models{1}{1,dimn}(:,rr)), models{1}{1,dimn}(:,rr), 40,'k')
        end
        %scatter(1:length(models{1}{1,dimn}(:,rr)), models{1}{1,dimn}(:,rr), 20,'k')
        %plot(models{1}{1,dimn}(:,rr), 'color','m','linewidth',1.2)
    end
end
legend('2std','Fit', 'True')

disp(['similarities: ', num2str(smlty_tot_noref,2)])
set(gcf,'position',[1921           1         633         961])


%% Pot Fit Offset
offset_fit = squeeze(vi_var.offset_mean(:,1,1,:,1));
offset_tru = squeeze(true_params.offset(:,1,1,:,1));

figure;
for ll = 1:size(offset_fit,2)
subplot(size(offset_fit,2),1,ll); hold on
    plot(offset_fit(:,ll))
    plot(offset_tru(:,ll))
end

figure; imagesc(squeeze(observed_data(:,1,1,:,1)))
colormap('gray');


%%
function vi_var = perturb_cp(vi_var, amp)

if nargin<2
    amp = 1;
end

% Get the amplitude of each CP-factors
[CP_normed, ~] = normalize_cp_std(vi_var.CP_mean,vi_var.CP_variance,0);

% Get CP = 0
CP_null = find(abs(CP_normed{end})<1e-12 );
CP_full = find(abs(CP_normed{end})>1e-12 );

% Get smallest CP
[~,locmin] = min(CP_normed{end}(CP_full));
CPmin = CP_full(locmin);

% Perturb null CP
CP_mean = vi_var.CP_mean;
CP_variance = vi_var.CP_variance;

%R = size(CP_mean{1},2);

for dimn = 1:size(CP_mean,2)
    % Perturbation amplitude
    %perturb_amp = amp*mean(abs(CP_mean{1,dimn}(:,CPmin)));
    %varianc_amp = amp*mean(vi_var.CP_variance{1,dimn}(:,CPmin+R*(CPmin-1)));
    
    for rr = CP_null
        
        
        %CP_mean{1,dimn}(:,rr) = perturb_amp*randn(size(CP_mean{1,dimn},1),1);
        %CP_variance{1,dimn}(:,rr+R*(rr-1)) = varianc_amp*abs(randn(size(CP_mean{1,dimn},1),1));
        
        permute_factor = randperm(size(CP_mean{1,dimn},1));
        CP_mean{1,dimn}(:,rr) = amp*CP_mean{1,dimn}(permute_factor,CPmin);
        CP_variance{1,dimn}(:,rr) = amp^2*CP_variance{1,dimn}(permute_factor,CPmin);
  
    end
end

% Store
vi_var.CP_mean = CP_mean;
vi_var.CP_variance = CP_variance;

end

function CPa = augment_cp(CP,r)
% Add extra Components to CP for similarity comparisons

CPa = cell(1,length(CP));
for dimn = 1:length(CP)
    CPa{1,dimn} = zeros(size(CP{dimn},1),size(CP{dimn},2)+r);
    CPa{1,dimn}(:,1:size(CP{dimn},2)) = CP{dimn};
end

end

function CPr = reduce_cp(CP,rmid)

factor_new = 1:size(CP{1},2);
factor_new(rmid) = [];
CPr = cell(1,length(CP));
for dimn = 1:length(CP)
    CPr{1,dimn} = CP{1,dimn}(:,factor_new);
end

end



function offsets = init_offsets(Xdims, fit_offset_dim)
% Init Offset

%offsets_tmp = 0.01*randn(fit_offset_dim.*Xdims+not(fit_offset_dim));
offsets_tmp = 0*rand(fit_offset_dim.*Xdims+not(fit_offset_dim));
offsets_tmp = randn(fit_offset_dim.*Xdims+not(fit_offset_dim));
offsets     = repmat(offsets_tmp, fit_offset_dim + not(fit_offset_dim).*Xdims);

end


%% Linear Response correction
%disp('Linear Response correction...'); tic;
%vi_var_up = vi_update_linear_response(Xobs,vi_var,vi_param);
%tmp = toc;
%disp(['Linear Response correction... Done: ', num2str(tmp,2), 's'])
%plot_cp(vi_var_up.CP_mean, vi_var_up.CP_variance)

%plot_cp(vi_var_with_ard.CP_mean, vi_var_with_ard.CP_variance)
%plot_cp(vi_var_no_ard.CP_mean, vi_var_no_ard.CP_variance)
