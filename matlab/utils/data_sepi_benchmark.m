%% Load and organize data

% Add master folders
addpath(genpath('~/Documents/MATLAB/tensor_decomp'))

addpath('./data_analysis/')
addpath('./tensor_gcp/')
addpath('./tensor_vi/')
addpath('./tensorfact_master/')
addpath('./utils/')


%data_folder = '/nfs/gatsbystor/hugos/data_sepi_all/';
%resu_folder = '/nfs/gatsbystor/hugos/';

data_folder = '~/Documents/Data/data_sepi_all/';
resu_folder = '~/Documents/Data/data_sepi_all/';

% Dataset Parameters
flip_back_data = 0;
stitch_data    = 0;

% Which dataset to use: 'mismatch', 'standard'
type_data = 'standard';

% Load
name_data = [data_folder,'data_sepi_',type_data, '_stitching', num2str(stitch_data), '_flipback', num2str(flip_back_data)];
load(name_data)

% Keep regions from : 'N/A','RSPd','RSPg','SC','SUB','V1', 'Hip'
kept_region = {'RSPd','RSPg','SC'};

% Attribute to group neurons: 'ctype','layer','region'
neuron_group_discriminant = 'layer';

% Load
data_sepi = data_sepi_load(data_sepi,kept_region,neuron_group_discriminant);

% Extract experimental parameters
param_names = fieldnames(data_sepi);
for parami = 1:length(param_names)
    eval([param_names{parami} '=data_sepi.' param_names{parami},';']);
end

Xobs = observed_tensor;

% Use a 3D or 4D tensor
collapse_conditions = 1;

% observed_tensor dim
if collapse_conditions
    final_dim = 3;
else 
    final_dim = ndims(Xobs);
end
experimental_parameters = data_sepi;

% Reduce Dataset and keep a single animal experiment
dim_neurons = 1;
dim_session = 4;
dim_other = 1:ndims(observed_tensor);
dim_other([dim_neurons,dim_session]) = [];

keep_session = [10] ;
remo_session = 1:size(observed_data,dim_session);
remo_session(keep_session) = [];


observed_mat = permute(observed_data, [dim_neurons,dim_session,dim_other]);
observed_mat = observed_mat(:,:,1);

keep_neuron = find(sum(observed_mat(:,keep_session),2));
remo_neuron = find(1-sum(observed_mat(:,keep_session),2));


observed_tensor = tensor_remove(observed_tensor,dim_session,remo_session);
observed_tensor = tensor_remove(observed_tensor,dim_neurons,remo_neuron);

observed_data   = tensor_remove(observed_data,  dim_session,remo_session);
observed_data   = tensor_remove(observed_data,  dim_neurons,remo_neuron);

direction = direction(:,keep_session,:);

record_region   = record_region(keep_neuron,:);
record_layer    = record_layer(keep_neuron,:);
record_celltype = record_celltype(keep_neuron,:);

neuron_group{1} = neuron_group{1}(keep_neuron,:);
sum(neuron_group{1} )
dict_region(record_region)

% Summarized the kept neurons

figure;
histogram(record_region)
xticks(unique(record_region))
xticklabels(cellstr(dict_region)')



%%







%% Fit
R = 7;

fit_offset_dim =1*[1,0,1,1];
Xobs = squeeze(observed_tensor);
observed_data = 1;

% Fit parameters
vi_param = struct();
vi_param.ite_max = 2000;
vi_param.observed_data = observed_data;
vi_param.fit_offset_dim = fit_offset_dim;
vi_param.shared_precision_dim= 0*[0,1,1,1];
vi_param.dim_neuron= 0;
vi_param.neurons_groups = neuron_group{1};
vi_param.update_CP_dim = ones(1,ndims(Xobs));
vi_param.R = R;

vi_param.shape_update = 'MM-G';

ninit = 1;


vi_var_tot = cell(1,ninit);
%parfor nn = 1:ninit 
for nn = 1:ninit 
    disp(['Current init: ', num2str(nn)])
    vi_var = struct();
    vi_var.shape = 8;
    vi_var = tensor_variational_inference(Xobs,vi_param,vi_var);
    
    vi_var_tot{nn} = vi_var;
end

% 
% 
% filename = [resu_folder,'vi_fit',type_data, '_stitching', num2str(stitch_data), '_flipback', num2str(flip_back_data),'_collapsed', num2str(collapse_conditions),'_',cell2mat(kept_region),'_' ,datestr(now,'yyyy_mm_dd_HH_MM')];
% save(filename,'vi_var_tot','vi_param','observed_tensor','-v7.3')
% 
% 
% 
% 





%%

% 
% % 
% [mean,variance] = squeeze_missing(...
%     vi_var.CP_mean,...
%     vi_var.CP_variance,vi_param.observed_data, [1,4]);
% 
% 
% 
% plot_cp(mean,variance)



