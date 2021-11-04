
% Add master folders
addpath('./data_analysis/')
addpath('./tensor_gcp/')
addpath('./tensor_vi/')
addpath('./utils/')

addpath(genpath('../tensor-demo-master/matlab/'))
addpath(genpath('../L-BFGS-B-C-master/'))


data_folder = '/nfs/gatsbystor/hugos/data_sepi_all/';
resu_folder = '/nfs/gatsbystor/hugos/';

%data_folder = '~/Documents/Data/data_sepi_all/'; 
%resu_folder = '~/Documents/Data/data_sepi_all/';

%data_folder = './../NEURIPS/';
%resu_folder = './../NEURIPS/';

% Dataset Parameters
flip_back_data = 0;
stitch_data    = 0;

% Which dataset to use: 'mismatch', 'standard'
type_data = 'standard';

% Load
name_data = [data_folder,'data_sepi_',type_data, '_stitching', num2str(stitch_data), '_flipback', num2str(flip_back_data)];
load(name_data)

% Keep regions from : 'N/A','RSPd','RSPg','SC','SUB','V1', 'Hip'
kept_region = {'RSPd','RSPg'};

% Attribute to group neurons: 'ctype','layer','region'
neuron_group_discriminant = 'layer';

% Load
data_sepi = data_sepi_load(data_sepi,kept_region,neuron_group_discriminant);

% Extract experimental parameters
param_names = fieldnames(data_sepi);
for parami = 1:length(param_names)
    eval([param_names{parami} '=data_sepi.' param_names{parami},';']);
end

experimental_parameters = data_sepi;

%%

observed_tensor = experimental_parameters.observed_tensor;
observed_dims = size(observed_tensor);
k_trials = size(observed_tensor,5);

observed_data = 1;

% Sizes of Train/Tests sets, Number of LNO
k_test = k_trials/2;         % Size of the test set
k_max  = 24;                 % Number of cross validation folders
k_max  = 72; 

% Get Test/Train Folders and random neuron index
% Get outer folder ids
ids = 1:k_trials;
[test_folder, train_folder] = xval_idx(ids,k_test,k_max);

k_folder = size(train_folder,1);

rank_tot = [1,2,4,6,8,10,12,14];
k_rank = length(rank_tot);


experimental_parameters.test_folder  = test_folder;
experimental_parameters.train_folder = train_folder;

%% Fit parameters for tensor_vi

ite_max = 10000; 

vi_param0 = struct();
vi_param0.ite_max = ite_max; 
vi_param0.observed_data = observed_data;
vi_param0.fit_offset_dim = [1,0,1];
vi_param0.shared_precision_dim= 1*[0,1,1];
vi_param0.dim_neuron= 1;
vi_param0.neurons_groups = neuron_group{1};
vi_param0.update_CP_dim = ones(1,3);
vi_param0.shape_update = 'MM-G';

vi_param0.disppct = 50;

vi_var0 = struct();
vi_var0.shape = 20;
vi_var0.prior_a_mode = 100;
vi_var0.prior_b_mode = 1;
vi_var0.prior_a_shared = 100;
vi_var0.prior_b_shared = 1;

% 1 -> VI
% 2 -> CP
% 3 -> GCP
model_str = {'VI','CP','GCP'};

% Store CP models
factors_tot = cell(1,k_folder);
varianc_tot = cell(1,k_folder);
variabl_tot = cell(1,k_folder);

% Store Variance and Deviance Explained
E0_train_tot = cell(1,k_folder); D0_train_tot = cell(1,k_folder);
E0_tests_tot = cell(1,k_folder); D0_tests_tot = cell(1,k_folder);

E_train_tot = cell(1,k_folder); D_train_tot = cell(1,k_folder);
E_tests_tot = cell(1,k_folder); D_tests_tot = cell(1,k_folder);

parfor folder_id = 1:k_folder    
    % Test/Train
    cur_tests_folders  = test_folder( folder_id,:);
    cur_train_folders = train_folder(folder_id,:);
    
    Xobs_train = sum(sum(observed_tensor(:,:,:,:,cur_train_folders),4),5);
    Xobs_tests = sum(sum(observed_tensor(:,:,:,:,cur_tests_folders),4),5);
    
    factors_folder = cell(3,k_rank);
    varianc_folder = cell(1,k_rank);
    variabl_folder = cell(1,k_rank);
    E_train_folder = cell(3,k_rank);
    E_tests_folder = cell(3,k_rank);
    D_train_folder = cell(3,k_rank);
    D_tests_folder = cell(3,k_rank);
    
    
    for R_id = 1:k_rank
        disp(['Folder: ', num2str(folder_id), '/', num2str(k_folder),...
              ' Param:', num2str(R_id), '/', num2str(k_rank)])
        
        % Current Rank being tested
        R = rank_tot(1,R_id);
        
        %% Fit vi decomposition
        vi_param = vi_param0; vi_var = vi_var0;
        vi_param.R =R;
        % Fit
        vi_var = tensor_variational_inference(Xobs_train,vi_param,vi_var);
        % Reconstruct
        Xhat_VI = vi_var.shape.*exp(tensor_reconstruct(vi_var.CP_mean) + vi_var.offset_mean);
        % Store factors
        factors_folder{1,R_id} = vi_var.CP_mean;
        varianc_folder{1,R_id} = vi_var.CP_variance;
        variabl_folder{1,R_id} = vi_var;
        
        %% Fit CP decomposition
        CP_factors_rk = cp_als(tensor(Xobs_train),R,'maxiters',ite_max,'printitn',ite_max);
        % Grasp
        CP_factors_tmp = CP_factors_rk.U(:)';
        CP_lambdas_tmp = CP_factors_rk.lambda';
        CP_factors_rk = CP_factors_tmp;
        CP_factors_rk{1,4} = CP_lambdas_tmp;
        CP_factors_rk = absorb_normalizer(CP_factors_rk,3);
        % Reconstruct
        Xhat_CP = tensor_reconstruct(CP_factors_rk);
        % Possible negative values...
        Xhat_CP_pos = Xhat_CP; Xhat_CP_pos(find(Xhat_CP_pos<0)) = 0; 
        % Store factors
        factors_folder{2,R_id} = CP_factors_rk;
        
        %% Fit GCP decomposition
        GCP_factors_rk = gcp_opt(tensor(Xobs_train),R, 'func',@(x,m) exp(m)-x.*m , 'grad', @(x,m) exp(m)-x,'lower',-Inf,'maxiters',ite_max,'printitn',ite_max);
        % Grasp
        GCP_factors_tmp = GCP_factors_rk.U(:)';
        GCP_lambdas_tmp = GCP_factors_rk.lambda';
        GCP_factors_rk = GCP_factors_tmp;
        GCP_factors_rk{1,4} = GCP_lambdas_tmp;
        GCP_factors_rk = absorb_normalizer(GCP_factors_rk,3);
        % Reconstruct
        Xhat_GCP = exp(tensor_reconstruct(GCP_factors_rk));
        % Store factors
        factors_folder{3,R_id} = GCP_factors_rk;
        
        
        %% Basline SS and DE0
        Xhat_0 = mean(Xobs_train(:)).*ones(size(Xobs_train));
        E0_train_tot{1,folder_id} = sum((Xhat_0(:)-Xobs_train(:)).^2);
        E0_tests_tot{1,folder_id} = sum((Xhat_0(:)-Xobs_tests(:)).^2);
        
        D0_train_tot{1,folder_id} = deviance_poisson(Xobs_train, Xhat_0);
        D0_tests_tot{1,folder_id} = deviance_poisson(Xobs_tests, Xhat_0);
        
        
        % VI SS and DE
        E_train_folder{1,R_id} = sum((Xhat_VI(:)-Xobs_train(:)).^2);
        E_tests_folder{1,R_id} = sum((Xhat_VI(:)-Xobs_tests(:)).^2);
        D_train_folder{1,R_id}  = deviance_poisson(Xobs_train, Xhat_VI);
        D_tests_folder{1,R_id}  = deviance_poisson(Xobs_tests, Xhat_VI);
        
        
        % CP SS and DE
        E_train_folder{2,R_id} = sum((Xhat_CP(:)-Xobs_train(:)).^2);
        E_tests_folder{2,R_id} = sum((Xhat_CP(:)-Xobs_tests(:)).^2);
        D_train_folder{2,R_id}  = deviance_poisson(Xobs_train, Xhat_CP_pos);
        D_tests_folder{2,R_id}  = deviance_poisson(Xobs_tests, Xhat_CP_pos);
        
        
        % GCP SS and DE
        E_train_folder{3,R_id} = sum((Xhat_GCP(:)-Xobs_train(:)).^2);
        E_tests_folder{3,R_id} = sum((Xhat_GCP(:)-Xobs_tests(:)).^2);
        D_train_folder{3,R_id}  = deviance_poisson(Xobs_train, Xhat_GCP);
        D_tests_folder{3,R_id}  = deviance_poisson(Xobs_tests, Xhat_GCP);
       
        
        
        
 
        
        
    end
    
    factors_tot{1,folder_id} = factors_folder;
    varianc_tot{1,folder_id} = varianc_folder;
    variabl_tot{1,folder_id} = variabl_folder;
    E_train_tot{1,folder_id} = E_train_folder;
    E_tests_tot{1,folder_id} = E_tests_folder;
    D_train_tot{1,folder_id} = D_train_folder;
    D_tests_tot{1,folder_id} = D_tests_folder;
    
    
end

filename = [resu_folder,'benchmark_vi_cp_gcp_data_sepi_' ,datestr(now,'yyyy_mm_dd_HH_MM')];
save(filename,'factors_tot','varianc_tot','variabl_tot',...
    'E_train_tot','E_tests_tot','D_train_tot','D_tests_tot',...
    'E0_train_tot','E0_tests_tot','D0_train_tot','D0_tests_tot',...
    'experimental_parameters','vi_param0', '-v7.3')




