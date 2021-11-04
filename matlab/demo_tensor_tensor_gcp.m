%% GCP Tensor Decomposition of Count Data with constraints
addpath(genpath('./'))

%% Generate Dataset
% Model
add_offset  = 1;
add_missing = 0;
model_true  = 'poisson';

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

%% Fit Parameters

% Neuron-type constraint
Pg = cell(1,ndims(Xobs));
Pg{1} = neurons_groups;

% Iterations and gradient steps
ite_max = 2000; 
rho_max = 1e-8;
rho_min = 5e-1;
tau     = 1000;
period  = ite_max/200; 
etaf    = get_step(ite_max, rho_max,rho_min,period,tau);
figure; plot(etaf); title('ADAM gradient steps'); xlabel('Iterations')

% Gradient steps 
rho_offset     = etaf;
rho_decomp     = 0.1*etaf;

% Gather in Structure
fit_param =struct();
fit_param.Pg = Pg;
fit_param.model = 'poisson';
fit_param.disppct = 10;
fit_param.ite_max = ite_max;
fit_param.rho_offset = rho_offset;
fit_param.rho_decomp = rho_decomp;
fit_param.fit_decomp_dim = ones(1,ndims(Xobs));
fit_param.fit_offset_dim = fit_offset_dim;

% Optimizer
opt = 'ADAMNC';
fit_param.beta1 = 0.9;
fit_param.beta2 = 0.999;
fit_param.opt   = opt;

% Tensor Rank
R = 4;
fit_param.R       = R;

% Neuron group penalty factor
fit_param.lambdag = 0.1.* sqrt(numel(Xobs)).*ones(size(fit_param.Pg{1,1},2),fit_param.R ); 

%1st Pass
results = tensor_mrp(Xobs,fit_param);

%% Plot Results

% Compare fit and simulation
if R > Rtrue
    CPt = augment_cp(CPtrue,R-Rtrue);
else
    CPt = CPtrue;
end

models = {CPt, results.fit.CP};

% Estimate Similarity Fit/True
[smlty_tot_noref,~,permt_tot,~,sig_tot] = ...
    get_similarity_tot(models,ones(1,ndims(Xobs)),1);

% Use it to reorder factors
models = reorder_cps(models,permt_tot,sig_tot);



% Plot
plot_cp(models{2})
for dimn=1:size(models{1},2)
    for rr = 1:size(models{1}{1},2)
        subplot(size(models{1}{1},2), size(models{1},2), (rr-1)*size(models{1},2) + dimn)
        
        if dimn <3
            scatter(1:length(models{1}{1,dimn}(:,rr)), models{1}{1,dimn}(:,rr), 10,'m', 'filled')
        else
            scatter(1:length(models{1}{1,dimn}(:,rr)), models{1}{1,dimn}(:,rr), 40,'m', 'filled')
            scatter(1:length(models{1}{1,dimn}(:,rr)), models{1}{1,dimn}(:,rr), 40,'k')
        end
    end
end
legend('Fit', 'True')






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

