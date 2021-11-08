%% Probabilistic Tensor Decomposition of Count Data
%       - Generate toydataset of rank R = 4 with missing Data and Offset
%       - Illustrate ARD for tensor rank selection 
%       - Plot Posteriors estimate
addpath(genpath('./'))

%% Generate Dataset
% Negative binomial model with missing observation and an offset
add_offset  = 1; 
add_missing = 1;
model_true  = 'negative_binomial';

% Observed Tensor Dimensions
Xdims = [100,70,3,4,5];

% True Rank
Rtrue = 4;

% For Reproduction purposes
rng(1)

% Generate Toy Dataset
[Xobs,observed_data,true_params] = ...
    build_toydataset(model_true,Rtrue,Xdims,add_offset,add_missing);

% Grasp simulated parameters
param_names = fieldnames(true_params);
for parami = 1:length(param_names)
    eval([param_names{parami} '=true_params.' param_names{parami},';']);
end

% Plot True Factors 
plot_cp(true_params.CPtrue)

%% Variational Inference
clc
R = 6;

% Fit parameters
vi_param = struct();

% Test rank
vi_param.R = R;

% Variational EM steps
vi_param.ite_max = 4000;

% Observed/Missing data
vi_param.observed_data = observed_data;

% Constrained Offset dimensions
vi_param.fit_offset_dim = add_offset*fit_offset_dim;

% ARD like parameters
vi_param.dim_neuron= 1;
vi_param.neurons_groups = neurons_groups;
vi_param.shared_precision_dim= [0,1,1,1,1];
vi_param.update_CP_dim = ones(1,ndims(Xobs));
vi_param.shape_update = 'MM-G';

% Increase speed by exploiting missing data structure 
if add_missing
    vi_param.sparse = 'block';
else
    vi_param.sparse = 'false';
end

% Shared initialization
vi_var0 = struct();
vi_var0.shape = 120;
vi_param.disppct = 0.1;

% Fit With ARD
[vi_var0,vi_param0] = vi_init(Xobs, vi_param, vi_var0);
vi_var_with_ard = tensor_variational_inference(Xobs,vi_param,vi_var0);

% Fit Without ARD
vi_param_no_ard = vi_param;
vi_param_no_ard.shared_precision_dim= 0*[0,1,1,1,1];
vi_param_no_ard.dim_neuron= 0;
vi_var_no_ard = tensor_variational_inference(Xobs,vi_param_no_ard,vi_var0);

%% Choose Method to plot
vi_var = vi_var_with_ard;
%vi_var = vi_var_no_ard;

%% Plot Fit summary
figure
subplot(1,2,1); hold on
plot(1:vi_param.ite_max, shape*ones(vi_param.ite_max,1), 'color','m', 'linewidth',2,'linestyle','--')
plot(1:vi_param.ite_max, vi_var.shape_tot, 'color','k', 'linewidth',2)
box on; xlabel('Iteration'); title('Shape Parameter')
%ylim([0, 200])

subplot(1,2,2)
plot(1:vi_param.ite_max, vi_var.loss_tot, 'color','k', 'linewidth',2)
box on; xlabel('Iteration'); title('Approximate FE')
set(gcf,'position',[1921         340         635         219])

%% Plot Fitted Factors
% Compare fit and simulation
R = vi_param.R;
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


%% Plot Fitted Offset 

offset_true = squeeze(true_params.offset(:,1,1,:,1));
offset_fit_mean = squeeze(vi_var.offset_mean(:,1,1,:,1));
offset_fit_vari = squeeze(vi_var.offset_variance(:,1,1,:,1));

figure;
for ll = 1:size(offset_fit_mean,2)
    
    offset_true_cur = offset_true(:,ll);
    offset_fit_mean_cur = offset_fit_mean(:,ll);
    offset_fit_variance_cur = offset_fit_vari(:,ll);
    
    x_cur = 1:length(offset_true_cur);
    
    up = offset_fit_mean_cur + 1*sqrt(abs(offset_fit_variance_cur));
    lo = offset_fit_mean_cur - 1*sqrt(abs(offset_fit_variance_cur));
    
    subplot(size(offset_fit_mean,2),1,ll); hold on
    
    % Patch fit std intervals
    patch([x_cur(:); flipud(x_cur(:))]', [up(:); flipud(lo(:))]',...
        'k', 'FaceAlpha',0.2,'EdgeAlpha',0)
    
    % Plot fit
    plot(x_cur, offset_fit_mean_cur, 'k', 'linewidth',1.3)
    
    % Plot True
    plot(x_cur, offset_true_cur, 'm', 'linewidth',1.3)
    
    if ll ==1
        title('Fit Offset')
    elseif ll == size(offset_fit_mean,2)
        xlabel('Neuron Dim')
    end
    
    ylabel(['Session. ', num2str(ll)])
    box on; axis tight;
    
end

%% Plot Missing data structure
figure; 
imagesc(squeeze(observed_data(:,1,1,:,1)))
xlabel('Neuron Dim')
ylabel('Session Dim')
colormap('gray');

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
