%% Probabilistic Tensor Decomposition of Count Data
% Illustrate ARD for tensor rank (2/2): Rank choice comparison
addpath(genpath('./'))
addpath(genpath('./../L-BFGS-B-C-master/'))
addpath(genpath('./../tensor-demo-master/'))
%% Generate Dataset
% Model
add_offset  = 0;
add_missing = 0;
model_true  = 'negative_binomial';

% Observed Tensor Dimensions
Xdims = [100,70,3,4,5];

% True Rank
Rtrue = 4;

% For Reproduction purposes
rng(100)

% Simulate Toy Dataset
[Xobs,observed_data,true_params, Xtrue] = ...
    build_toydataset(model_true,Rtrue,Xdims,add_offset,add_missing);

% Grasp simulated parameters
param_names = fieldnames(true_params);
for parami = 1:length(param_names)
    eval([param_names{parami} '=true_params.' param_names{parami},';']);
end

% Plot True dataset 
plot_cp(true_params.CPtrue)

%% Fit Variational Inference
clc
R = 6;

% Fit parameters
vi_param = struct();
vi_param.ite_max = 4000;
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
vi_param.disppct = 10;


Ntot = 20;
results_vi = cell(Ntot,1);
parfor nn=1:Ntot
    % Shared initialization
    vi_var0 = struct();
    vi_var0.shape = 120;
    
    [vi_var0,vi_param0] = vi_init(Xobs, vi_param, vi_var0);
    vi_var_with_ard = tensor_variational_inference(Xobs,vi_param,vi_var0);
    results_vi{nn} = vi_var_with_ard
end


%% Fit GCP-NB decomposition
rank_tot = 1:8;
shap_tot = [1,10,80,200];
ite_max = 4000;
results_gcpnb = cell(Ntot,length(rank_tot),length(shap_tot));

for nn=1:Ntot
    for shid = 1:length(shap_tot)
        shape_nb_cur = shap_tot(shid);
        
        parfor rid = 1:length(rank_tot)
            
            Rcur = rank_tot(rid)
            factors_rk = gcp_opt(tensor(Xobs), Rcur,...
                'func',@(x,m) (shape_nb_cur+x).*log(1+exp(m))-x.*m ,...
                'grad', @(x,m) (shape_nb_cur+x)./(1+exp(-m))-x,...
                'lower',-Inf,'maxiters',ite_max,'printitn',ite_max);
            
            % Grasp
            factors_tmp = factors_rk.U(:)';
            lambdas_tmp = factors_rk.lambda';
            factors_rk = factors_tmp;
            factors_rk{1,6} = lambdas_tmp;
            factors_rk = absorb_normalizer(factors_rk,3);
            % Reconstruct
            Xhat = shape_nb_cur*exp(tensor_reconstruct(factors_rk));
            
            resultsc = struct();
            resultsc.CP = factors_rk;
            resultsc.Xhat = Xhat;
            results_gcpnb{nn,rid,shid} = resultsc;
            
        end
    end
end

%% Process VI and GCP-NB

% Process VI
s2dif_vi = zeros(Ntot,1);
rankd_vi = zeros(Ntot,1);
model_vi =  cell(Ntot,1); 
varia_vi =  cell(Ntot,1); 
for nn=1:Ntot
   % ARD threshold to detect number of components
   norm_tot = normalize_cp(results_vi{nn}.CP_mean);
   norm_tot = norm_tot{6};
   Rc = length(find(norm_tot>0.1));
   rankd_vi(nn) = Rc;
   
   % Grasp CP factors and reconstruct Xhat
   Xhat = exp(tensor_reconstruct(results_vi{nn}.CP_mean))*results_vi{nn}.shape;
   
   % Store
   model_vi{nn} = results_vi{nn}.CP_mean;
   varia_vi{nn} = results_vi{nn}.CP_variance;
   s2dif_vi(nn) = sum((Xhat(:) - Xtrue(:)).^2);

end
[sim_within_vi,ref_vi] = get_similarity_tot(model_vi');

% Process GCP-NB
s2dif_gcpnb = zeros(Ntot,length(rank_tot),length(shap_tot));
model_gcpnb =  cell(Ntot,length(rank_tot),length(shap_tot));
for nn=1:Ntot
    for rid = 1:length(rank_tot)
        for shid = 1:length(shap_tot)
            
            Xhat_tmp = results_gcpnb{nn,rid,shid}.Xhat(:);
            Xhat_tmp(Xhat_tmp>1e4)=0;
            
            model_gcpnb{nn,rid,shid} = results_gcpnb{nn,rid,shid}.CP;
            s2dif_gcpnb(nn,rid,shid) = sum((Xhat_tmp(:) - Xtrue(:)).^2);
        end
    end
    
end
[sim_within_gcpnb,ref_tot] = get_similarity_tot(permute(model_gcpnb,[3,2,1]));

%% Plots

% Choice of Hyperparameters for GCP
Rref_gcp_1 = 4;
Sref_gcp_1 = 3;
Nref_gcp_1 = ref_tot(Sref_gcp_1,Rref_gcp_1);

Rref_gcp_2 = 5;
Sref_gcp_2 = 3;
Nref_gcp_2 = ref_tot(Sref_gcp_2,Rref_gcp_2);

% Colors
color_gcpnb = [...
    linspace(0,1,length(shap_tot))',...
    linspace(0,0,length(shap_tot))',...
    linspace(0,0,length(shap_tot))'];

% Plots
figure

% Goodness-of-Fit
subplot(1,2,1); hold on
% All GCP-NB
for shid = 1:length(shap_tot)
    plot(rank_tot,squeeze(mean(s2dif_gcpnb(:,:,shid),1)), 'linewidth',2,'color', color_gcpnb(shid,:))
end
% VI-GCP
plot(rankd_vi, s2dif_vi,'color', [0 0.5 1], 'linewidth',2)
scatter(rankd_vi, s2dif_vi,150, [0 0.5 1], 'filled')
% Ref 1-2 GCP-NB
scatter(rank_tot(Rref_gcp_1), s2dif_gcpnb(Nref_gcp_1, Rref_gcp_1,Sref_gcp_1),80, color_gcpnb(Sref_gcp_1,:), 'filled')
scatter(rank_tot(Rref_gcp_2), s2dif_gcpnb(Nref_gcp_2, Rref_gcp_1,Sref_gcp_2),80, color_gcpnb(Sref_gcp_2,:), 'filled')
title('Goodness-of-Fit')
box on; axis tight; xlabel('R')
for shid = 1:length(shap_tot)
    titletot{1,shid} = ['NB(r=',num2str(shap_tot(shid)),')'];
end
titletot{1,shid+1} = 'VB-GCP';
box on; axis tight; xlabel('R')
legend(titletot)
set(gcf,'position', [593         369        687         257])

% Similarities
subplot(1,2,2); hold on
% All GCP-NB
for shid = 1:length(shap_tot)
    plot(rank_tot,mean(sim_within_gcpnb(shid,:,:),3), 'linewidth',2,'color', color_gcpnb(shid,:))
end
% VI-GCP
scatter(rankd_vi(2:end), sim_within_vi,80, [0 0.5 1], 'filled')
% Ref 1-2 GCP-NB
scatter(rank_tot(Rref_gcp_1), mean(sim_within_gcpnb(Sref_gcp_1,Rref_gcp_1,:)),80, color_gcpnb(Sref_gcp_1,:), 'filled')
scatter(rank_tot(Rref_gcp_2), mean(sim_within_gcpnb(Sref_gcp_2,Rref_gcp_2,:)),80, color_gcpnb(Sref_gcp_2,:), 'filled')

title('Similarities')
box on; axis tight; ylim([0,1.1]); xlabel('R')


%% Plot ref VI and GCP-NB

model_to_plot = {model_vi{ref_vi};...
    results_gcpnb{Nref_gcp_1,Rref_gcp_1,Sref_gcp_1}.CP;...
    results_gcpnb{Nref_gcp_2,Rref_gcp_2,Sref_gcp_2}.CP};

varia_to_plot = {varia_vi{ref_vi};[];[]};

for mid = 1:size(model_to_plot,1)
    
    model_cur = model_to_plot{mid};
    varia_cur = varia_to_plot{mid};
    
    R = size(model_cur{1},2);
    % Compare fit and simulation
    if R > Rtrue
        CPt = augment_cp(CPtrue,R-Rtrue);
    elseif R < Rtrue
        model_cur = augment_cp(model_cur,Rtrue-R);
    else
        CPt = CPtrue;
    end
    
    
    
    models = {CPt, model_cur};
    variances = {[],varia_cur};
    
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
    set(gcf,'position',[1921           1         633         160.2*R])
    
end










%% Helpers

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

