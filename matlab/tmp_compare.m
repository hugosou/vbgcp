%% Save
folder = '~/Documents/PYTHON/tests_matlab/';
CP = true_params.CPtrue;

save([folder, 'for_testing'],'Xobs', 'vi_param',  'vi_var_with_ard' ,'CP')

%%
cd ~/Documents/MATLAB/tensor_decomp/
clc
addpath(genpath('./'))
folder = '~/Documents/PYTHON/tests_matlab/';
%load([folder, 'for_testing'])
vi_param.sparse = 'false';
vi_param.ite_max = 100;
%vi_var_with_ard.shape



%[vi_var0,vi_param0] = vi_init(Xobs, vi_param, vi_var0);
vi_var_with_ard = tensor_variational_inference(Xobs,vi_param,vi_var_with_ard);

%plot_cp(vi_var_with_ard.CP_mean)

save([folder, 'for_testing2'],'Xobs', 'vi_param',  'vi_var_with_ard','CP')


%%
get_similarity(models,1)
get_similarity({CPt, vi_var_with_ard.CP_mean}, 1)

%%




costMat = [[4, 1, 3, 20, 32.2]; [2, 0, 5, 78, 8]; [3, 2, 2, 45, 69]; [54,7,1,2,3]; [0,1,2,3,4]];



[assignment,cost] = munkres(costMat);


aa =  [[ 2.07828420e-01,  1.15104473e-04,  3.31740712e-04, -1.78899681e-04,...
   0.00000000e+00,  0.00000000e+00];
 [-2.00823346e-04,  4.48106447e-02,  2.19198183e-06,  2.65468653e-05,...
   0.00000000e+00,  0.00000000e+00];
 [ 8.81234431e-04, -1.60403384e-06,  1.78256294e-03,  3.17978795e-05,...
   0.00000000e+00,  0.00000000e+00];
 [ 1.65597709e-03, -5.44078799e-06,  1.09505881e-04, -1.34436253e-03,...
   0.00000000e+00,  0.00000000e+00];
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,...
   0.00000000e+00,  0.00000000e+00];
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,...
   0.00000000e+00,  0.00000000e+00]];

   
[assignment,cost] = munkres(-aa);
   
   
assignment-1
   
%%

% With ARD
vi_param.ite_max = 10;
vi_var0 = struct();
vi_var0.shape = 85;
vi_param.disppct = 0.1;
[vi_var0,vi_param0] = vi_init(Xobs, vi_param, vi_var0);
vi_var_with_ard1 = tensor_variational_inference(Xobs,vi_param,vi_var0);


vi_var0 = struct();
vi_var0.shape = 85;
vi_param.disppct = 0.1;
[vi_var0,vi_param0] = vi_init(Xobs, vi_param, vi_var0);
vi_var_with_ard2 = tensor_variational_inference(Xobs,vi_param,vi_var0);


vi_var0 = struct();
vi_var0.shape = 85;
vi_param.disppct = 0.1;
[vi_var0,vi_param0] = vi_init(Xobs, vi_param, vi_var0);
vi_var_with_ard3 = tensor_variational_inference(Xobs,vi_param,vi_var0);


vi_var0 = struct();
vi_var0.shape = 85;
vi_param.disppct = 0.1;
[vi_var0,vi_param0] = vi_init(Xobs, vi_param, vi_var0);
vi_var_with_ard4 = tensor_variational_inference(Xobs,vi_param,vi_var0);


%%
clc
models = {vi_var_with_ard1.CP_mean,vi_var_with_ard2.CP_mean,vi_var_with_ard3.CP_mean,vi_var_with_ard4.CP_mean};


[smlty,perm_final,sign_final] = get_similarity(models, 1);


smlty

perm_final-1
%folder = '~/Documents/';
%save([folder, 'for_testing3'],'models')





%%











