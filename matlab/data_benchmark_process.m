%% Process benchmark analysis 
%load('./../NEURIPS/neurips_benchmark_vi_cp_gcp_data_sepi.mat')
load('./../NEURIPS/neurips_benchmark_vi_cp_gcp_data_sepi_with_gcpnb_baseline.mat')
addpath(genpath('./'))
%% Gather fits accross folders
Fmax = size(factors_tot,2);
fators_tot_f = factors_tot(1:Fmax) ;
E_train_tot_f = E_train_tot(1:Fmax);
E_tests_tot_f = E_tests_tot(1:Fmax);
D_train_tot_f = D_train_tot(1:Fmax);
D_tests_tot_f = D_tests_tot(1:Fmax);

% Reference
E0_train_tot_f = E0_train_tot(1:Fmax);
D0_train_tot_f = D0_train_tot(1:Fmax);
E0_tests_tot_f = E0_tests_tot(1:Fmax);
D0_tests_tot_f = D0_tests_tot(1:Fmax);


fators_tot_f_ordered = cell(...
    size(fators_tot_f{1},1),...
    size(fators_tot_f{1},2),...
    Fmax);

E_train_tot_ordered = zeros(...
    size(fators_tot_f{1},1),...
    size(fators_tot_f{1},2),...
    Fmax);

D_train_tot_ordered = zeros(...
    size(fators_tot_f{1},1),...
    size(fators_tot_f{1},2),...
    Fmax);

E_tests_tot_ordered = zeros(...
    size(fators_tot_f{1},1),...
    size(fators_tot_f{1},2),...
    Fmax);

D_tests_tot_ordered = zeros(...
    size(fators_tot_f{1},1),...
    size(fators_tot_f{1},2),...
    Fmax);


for folder_id = 1:Fmax
    
    fators_tot_f_ordered(:,:,folder_id)= fators_tot_f{1,folder_id};
    
    E_train_tot_ordered(:,:,folder_id) = cell2mat(E_train_tot_f{1,folder_id});
    D_train_tot_ordered(:,:,folder_id) = cell2mat(D_train_tot_f{1,folder_id});
    E_tests_tot_ordered(:,:,folder_id) = cell2mat(E_tests_tot_f{1,folder_id});
    D_tests_tot_ordered(:,:,folder_id) = cell2mat(D_tests_tot_f{1,folder_id});
    
end

%% Estimate similarities between discovered factors 

[smlty_tot_noref,ref_tot,permt_tot,smlty_tot,sig_tot] = ...
    get_similarity_tot(fators_tot_f_ordered,[1,1,1]);

avg_similarities = mean(smlty_tot_noref,3);
std_similarities = std(smlty_tot_noref,[],3);

%% Calculate Variance and Deviance Explained on test and train folders

k_stud = size(D_train_tot_ordered,1);
k_rank = size(D_train_tot_ordered,2);

E0_train_tot_ordered = permute(repmat(cell2mat(E0_train_tot_f)', [1,k_stud,k_rank]),[2,3,1]);
D0_train_tot_ordered = permute(repmat(cell2mat(D0_train_tot_f)', [1,k_stud,k_rank]),[2,3,1]);
E0_tests_tot_ordered = permute(repmat(cell2mat(E0_tests_tot_f)', [1,k_stud,k_rank]),[2,3,1]);
D0_tests_tot_ordered = permute(repmat(cell2mat(D0_tests_tot_f)', [1,k_stud,k_rank]),[2,3,1]);


VE_train_tot = 1-E_train_tot_ordered./E0_train_tot_ordered;
DE_train_tot = 1-D_train_tot_ordered./D0_train_tot_ordered;
VE_tests_tot = 1-E_tests_tot_ordered./E0_tests_tot_ordered;
DE_tests_tot = 1-D_tests_tot_ordered./D0_tests_tot_ordered;


%% Plots

model_str = {'VB-GCP','CP','GCP'};
orange      = [1 0.5 0];
blue        = [0 0.5 1];
green       = [0 0.6 0.3];

colors_tot = [blue;orange;green];

% Only temporary for NB
model_str = {'VI','CP','GCP-Poiss'};
for nshape = 1:length(shap_tot)
    model_str{3+nshape} = ['GVP-NB(',num2str(shap_tot(nshape),2) ,')'];
end
colors_tot = [colors_tot;
    [linspace(0,0,length(shap_tot))',linspace(1,0,length(shap_tot))',linspace(0,1,length(shap_tot))']];

rank_test  = cellfun(@(Z) size(Z{1},2),  fators_tot_f_ordered(2,:,1));

% Similarities Only
figure; hold on
for ll = 1:size(fators_tot_f{1},1)
    errorbar(rank_test,avg_similarities(ll,:),std_similarities(ll,:),...
        'linewidth',2,'color', colors_tot(ll,:))
end
legend(model_str)
box on; ylim([0,1])


figure
% Variance Explain Train
subplot(2,2,1); hold on
for ll = 1:size(fators_tot_f{1},1)
    var_cur = squeeze(VE_train_tot(ll,:,:));
    avg_cur = mean(var_cur,2);
    std_cur = std(var_cur,[],2);
    errorbar(rank_test,avg_cur,std_cur,...
        'linewidth',2,'color', colors_tot(ll,:))
end
title('VE train')
box on; ylim([0,1])

% Variance Explain Tests
subplot(2,2,2); hold on
for ll = 1:size(fators_tot_f{1},1)
    var_cur = squeeze(VE_tests_tot(ll,:,:));
    avg_cur = mean(var_cur,2);
    std_cur = std(var_cur,[],2);
    errorbar(rank_test,avg_cur,std_cur,...
        'linewidth',2,'color', colors_tot(ll,:))
end
title('VE test')
box on; ylim([0,1])

% Deviance Explain Train
subplot(2,2,3); hold on
for ll = 1:size(fators_tot_f{1},1)
    var_cur = squeeze(DE_train_tot(ll,:,:));
    avg_cur = mean(var_cur,2);
    std_cur = std(var_cur,[],2);
    errorbar(rank_test,avg_cur,std_cur,...
        'linewidth',2,'color', colors_tot(ll,:))
end
title('DE train')
box on; ylim([0,1])

% Deviance Explain Test
subplot(2,2,4); hold on
for ll = 1:size(fators_tot_f{1},1)
    var_cur = squeeze(DE_tests_tot(ll,:,:));
    avg_cur = mean(var_cur,2);
    std_cur = std(var_cur,[],2);
    errorbar(rank_test,avg_cur,std_cur,...
        'linewidth',2,'color', colors_tot(ll,:))
end
title('DE test')
legend(model_str)
box on; ylim([0,1])

%% Figure Paper (no NegBin Baseline)

figure
subplot(1,3,1); hold on
%for ll = 1:size(fators_tot_f{1},1)
for ll = 1:3 
    var_cur = squeeze(VE_tests_tot(ll,:,:));
    avg_cur = mean(var_cur,2);
    std_cur = std(var_cur,[],2);
    errorbar(rank_test,avg_cur,std_cur,...
        'linewidth',2,'color', colors_tot(ll,:))
end
title('Test VE')
legend(model_str, 'location', 'southeast')
box on; xlabel('R')
%axis tight

subplot(1,3,2); hold on
%for ll = 1:size(fators_tot_f{1},1)
for ll = 1:3
    var_cur = squeeze(DE_tests_tot(ll,:,:));
    avg_cur = mean(var_cur,2);
    std_cur = std(var_cur,[],2);
    errorbar(rank_test,avg_cur,std_cur,...
        'linewidth',2,'color', colors_tot(ll,:))
end
title('Test DE')
box on; xlabel('R')
%axis tight

subplot(1,3,3); hold on
%for ll = 1:size(fators_tot_f{1},1)
for ll = 1:3
    errorbar(rank_test,avg_similarities(ll,:),std_similarities(ll,:),...
        'linewidth',2,'color', colors_tot(ll,:))
    
    plot(rank_test,avg_similarities(ll,:),'linewidth',2,'color', colors_tot(ll,:))
end
title('Similarities')
box on; xlabel('R')
%axis tight
set(gcf,'position', [593         369        1030         257])




%% Only for NB Baseline
% //////////////////////////////////////////////////////////////////// 
% NB baseline added and selected with DE, Sim or Training Likelihood//
% ////////////////////////////////////////////////////////////////////

% Select GCP-NB based on Poisson Deviance Explained
[~,NB_shape_dev] = max(mean(DE_tests_tot(4:end,:,:),3),[],1);
% Select GCP-NB based on Similarity
[~,NB_shape_sim] = max(avg_similarities(4:end,:), [],1);


% Estimate training Log-Likelihood of GCP-NegBin

% Grasp back test and train folders
test_folder  = experimental_parameters.test_folder;
train_folder = experimental_parameters.train_folder;
k_folder = size(train_folder,1);

% Init NegBin log likelihoods
logL_nb_tot_train = zeros(length(shap_tot), k_rank, k_folder);
logL_nb_tot_tests = zeros(length(shap_tot), k_rank, k_folder);

for folder_id = 1:k_folder
    disp(k_folder)
    % Test/Train
    cur_tests_folders  = test_folder( folder_id,:);
    cur_train_folders  = train_folder(folder_id,:);
    
    Xobs_train = sum(sum(observed_tensor(:,:,:,:,cur_train_folders),4),5);
    Xobs_tests = sum(sum(observed_tensor(:,:,:,:,cur_tests_folders),4),5);
    
    
    for nshape = 1:length(shap_tot)
        % Current Shape Param
        shape_nb_cur = shap_tot(nshape);
        % NB distribution Normalizers
        normalizer_cur_train = nb_normalizer(Xobs_train,shape_nb_cur);
        normalizer_cur_tests = nb_normalizer(Xobs_tests,shape_nb_cur);
        
        for R_id = 1:k_rank
            % GCP factors
            GCPNB_factors_rk = factors_tot{1,folder_id}{3+nshape,R_id};
            % Reconstructed Low-Rank Tensor
            What = tensor_reconstruct(GCPNB_factors_rk);
            % log-likelihood
            logLc_train = logL_nb(Xobs_train,What,shape_nb_cur,normalizer_cur_train);            
            logLc_tests = logL_nb(Xobs_tests,What,shape_nb_cur,normalizer_cur_tests);            
            % Store
            logL_nb_tot_train(nshape, R_id, folder_id) = sum(logLc_train(:));
            logL_nb_tot_tests(nshape, R_id, folder_id) = sum(logLc_tests(:));
        end
    end
    
end


logL_nb_avg_train = mean(logL_nb_tot_train,3);
logL_nb_avg_tests = mean(logL_nb_tot_tests,3);

[~, shape_max_log_id_train] = max(logL_nb_avg_train, [], 1);
[~, shape_max_log_id_tests] = max(logL_nb_avg_tests, [], 1);

% Select GCP-NB based Training NB likelihood
NB_shape_lik = shape_max_log_id_train;

%% Gather Selected Models

NB_shape_tot = [NB_shape_dev; NB_shape_sim; NB_shape_lik];
VE_train_shape = zeros(3+size(NB_shape_tot,1),size(avg_similarities,2),Fmax);
VE_tests_shape = zeros(3+size(NB_shape_tot,1),size(avg_similarities,2),Fmax);
DE_train_shape = zeros(3+size(NB_shape_tot,1),size(avg_similarities,2),Fmax);
DE_tests_shape = zeros(3+size(NB_shape_tot,1),size(avg_similarities,2),Fmax);

avg_similarities_shape = zeros(3+size(NB_shape_tot,1),size(avg_similarities,2));
std_similarities_shape = zeros(3+size(NB_shape_tot,1),size(avg_similarities,2));

VE_train_shape(1:3,:,:) = VE_train_tot(1:3,:,:);
VE_tests_shape(1:3,:,:) = VE_tests_tot(1:3,:,:);
DE_train_shape(1:3,:,:) = DE_train_tot(1:3,:,:);
DE_tests_shape(1:3,:,:) = DE_tests_tot(1:3,:,:);

avg_similarities_shape(1:3,:) = avg_similarities(1:3,:);
std_similarities_shape(1:3,:) = std_similarities(1:3,:);

for rr = 1:size(avg_similarities,2)
    for shape_id = 1:size(NB_shape_tot,1)
        
        id_cur = NB_shape_tot(shape_id,rr);
        
        VE_train_shape(3+shape_id,rr,:) = VE_train_tot(3+id_cur,rr,:);
        VE_tests_shape(3+shape_id,rr,:) = VE_tests_tot(3+id_cur,rr,:);
        DE_train_shape(3+shape_id,rr,:) = DE_train_tot(3+id_cur,rr,:);
        DE_tests_shape(3+shape_id,rr,:) = DE_tests_tot(3+id_cur,rr,:);
        
        smlt_tmp = smlty_tot_noref(3+id_cur,rr,:);
        avg_similarities_shape(3+shape_id,rr) = mean(smlt_tmp);
        std_similarities_shape(3+shape_id,rr) = std(smlt_tmp);
    end
end
%% Supplementary Figure With NegBin

model_str = {'VB-GCP','CP','GCP-Poisson','GCP-NB(Dev)','GCP-NB(Sim)','GCP-NB(Lik)'};
blue        = [0 0.5 1];
orange      = [1 0.5 0];
green       = [0 0.6 0.3];
red         = [1 0.2 0];
grey        = [0.5 0.5 0.5];
black       = [0 0 0];

colors_tot = [blue;orange;green;red;grey;black];
rank_test  = cellfun(@(Z) size(Z{1},2),  fators_tot_f_ordered(2,:,1));

% Similarities Only
figure; hold on
for ll = 1:size(avg_similarities_shape,1)
    errorbar(rank_test,avg_similarities_shape(ll,:),std_similarities_shape(ll,:),...
        'linewidth',2,'color', colors_tot(ll,:))
end
legend(model_str)
box on; ylim([0,1])
title('Similarities')
xlabel('R')

figure
% Variance Explain Train
subplot(2,2,1); hold on
for ll = 1:size(avg_similarities_shape,1)
    var_cur = squeeze(VE_train_shape(ll,:,:));
    avg_cur = mean(var_cur,2);
    std_cur = std(var_cur,[],2);
    errorbar(rank_test,avg_cur,std_cur,...
        'linewidth',2,'color', colors_tot(ll,:))
end
title('VE train')
box on; xlabel('R')

% Variance Explain Tests
subplot(2,2,2); hold on
for ll = 1:size(avg_similarities_shape,1)
    var_cur = squeeze(VE_tests_shape(ll,:,:));
    avg_cur = mean(var_cur,2);
    std_cur = std(var_cur,[],2);
    errorbar(rank_test,avg_cur,std_cur,...
        'linewidth',2,'color', colors_tot(ll,:))
end
title('VE test')
box on; xlabel('R')

% Deviance Explain Train
subplot(2,2,3); hold on
for ll = 1:size(avg_similarities_shape,1)
    var_cur = squeeze(DE_train_shape(ll,:,:));
    avg_cur = mean(var_cur,2);
    std_cur = std(var_cur,[],2);
    errorbar(rank_test,avg_cur,std_cur,...
        'linewidth',2,'color', colors_tot(ll,:))
end
title('DE train')
box on; xlabel('R') 

% Deviance Explain Test
subplot(2,2,4); hold on
for ll = 1:size(avg_similarities_shape,1)
    var_cur = squeeze(DE_tests_shape(ll,:,:));
    avg_cur = mean(var_cur,2);
    std_cur = std(var_cur,[],2);
    errorbar(rank_test,avg_cur,std_cur,...
        'linewidth',2,'color', colors_tot(ll,:))
end
title('DE test')
legend(model_str)
box on; xlabel('R')

















% 
% 
% for rr = 1:size(avg_similarities,2)
%    DE_avg_GCPNB_dev(rr) = mean(DE_tests_tot(3+NB_shape_dev(rr),rr,:),3);
%    DE_avg_GCPNB_sim(rr) = mean(DE_tests_tot(3+NB_shape_sim(rr),rr,:),3);
%    DE_avg_GCPNB_lik(rr) = mean(DE_tests_tot(3+NB_shape_lik(rr),rr,:),3);
%    DE_std_GCPNB_dev(rr) = std(DE_tests_tot(3+NB_shape_dev(rr),rr,:),[],3);
%    DE_std_GCPNB_sim(rr) = std(DE_tests_tot(3+NB_shape_sim(rr),rr,:),[],3);
%    DE_std_GCPNB_lik(rr) = std(DE_tests_tot(3+NB_shape_lik(rr),rr,:),[],3);
%    
%    VE_avg_GCPNB_dev(rr) = mean(VE_tests_tot(3+NB_shape_dev(rr),rr,:),3);
%    VE_avg_GCPNB_sim(rr) = mean(VE_tests_tot(3+NB_shape_sim(rr),rr,:),3);
%    VE_avg_GCPNB_lik(rr) = mean(VE_tests_tot(3+NB_shape_lik(rr),rr,:),3);
%    VE_std_GCPNB_dev(rr) = std(VE_tests_tot(3+NB_shape_dev(rr),rr,:),[],3);
%    VE_std_GCPNB_sim(rr) = std(VE_tests_tot(3+NB_shape_sim(rr),rr,:),[],3);
%    VE_std_GCPNB_lik(rr) = std(VE_tests_tot(3+NB_shape_lik(rr),rr,:),[],3);
%     
%    sim_avg_GCPNB_dev(rr) = avg_similarities(3+NB_shape_dev(rr),rr);
%    sim_std_GCPNB_dev(rr) = std_similarities(3+NB_shape_dev(rr),rr);
%    sim_avg_GCPNB_sim(rr) = avg_similarities(3+NB_shape_sim(rr),rr);
%    sim_std_GCPNB_sim(rr) = std_similarities(3+NB_shape_sim(rr),rr);
%    sim_avg_GCPNB_lik(rr) = avg_similarities(3+NB_shape_lik(rr),rr);
%    sim_std_GCPNB_lik(rr) = std_similarities(3+NB_shape_lik(rr),rr);
%    
%     
% end
% 
% %%
% 
% 
% figure; 
% subplot(1,3,1);hold on
% % VE VB-GCP, CP, GCP Poisson
% for ll = 1:3
%     errorbar(rank_test,mean(VE_tests_tot(ll,:,:),3),std(VE_tests_tot(ll,:,:),[],3),...
%         'linewidth',2,'color', colors_tot(ll,:))
% end
% % VE GCP NB
% errorbar(rank_test,VE_avg_GCPNB_dev,VE_std_GCPNB_dev,...
%     'linewidth',2,'color', [0,0,0])
% errorbar(rank_test,VE_avg_GCPNB_sim,VE_std_GCPNB_sim,...
%     'linewidth',2,'color', [0.5,0.5,0.5])
% errorbar(rank_test,VE_avg_GCPNB_lik,VE_std_GCPNB_lik,...
%     'linewidth',2,'color', 'm')
% title('Test VE')
% legend({'VB-GCP','CP','GCP Poissson','GCP NB dev','GCP NB sim'}, 'location', 'southeast')
% box on; xlabel('R')
% 
% subplot(1,3,2);hold on
% % DE VB-GCP, CP, GCP Poisson
% for ll = 1:3
%     errorbar(rank_test,mean(DE_tests_tot(ll,:,:),3),std(DE_tests_tot(ll,:,:),[],3),...
%         'linewidth',2,'color', colors_tot(ll,:))
% end
% % DE GCP NB
% errorbar(rank_test,DE_avg_GCPNB_dev,DE_std_GCPNB_dev,...
%     'linewidth',2,'color', [0,0,0])
% errorbar(rank_test,DE_avg_GCPNB_sim,DE_std_GCPNB_sim,...
%     'linewidth',2,'color', [0.5,0.5,0.5])
% errorbar(rank_test,DE_avg_GCPNB_lik,DE_std_GCPNB_lik,...
%     'linewidth',2,'color', 'm')
% title('Test DE')
% box on; xlabel('R')
% 
% subplot(1,3,3);hold on
% % Similarities VB-GCP, CP, GCP Poisson
% for ll = 1:3
%     errorbar(rank_test,avg_similarities(ll,:),std_similarities(ll,:),...
%         'linewidth',2,'color', colors_tot(ll,:))
% end
% % Similarities VE
% errorbar(rank_test,sim_avg_GCPNB_dev,sim_std_GCPNB_dev,...
%     'linewidth',2,'color', [0,0,0])
% errorbar(rank_test,sim_avg_GCPNB_sim,sim_std_GCPNB_sim,...
%     'linewidth',2,'color', [0.5,0.5,0.5])
% errorbar(rank_test,sim_avg_GCPNB_lik,sim_std_GCPNB_lik,...
%     'linewidth',2,'color', 'm')
% title('Similarities')
% box on; xlabel('R')
% 
% set(gcf,'position', [593         369        1030         257])



function xf = nb_normalizer(x,shape)
x = x(:);
xx = repmat(x',[max(x),1]);

xf = zeros(max(x), length(x));
xf = xf + (1:max(x))'-1;
xf = log((xf+shape).*(xf<=(xx-1))+1.*(xf>(xx-1))) ;

xf = sum(xf,1)';
end

function logL = logL_nb(x,w,shape,normalizer)
x = x(:);
w = w(:);
logL = normalizer + x.*w - (x+shape).*log(1+exp(w));

end

