function Xpred = lno_test_tcp(results,idtest,idtrain, Xtest)

results_mlr = results.fit.t;
fit_param_mlr   = results.fit_param.t;

Xtest_mi = tensor_remove(Xtest,1,idtest);



% Grasp MLR train fit
Uk = results.fit.cp.Hosvd;
Nhat_train  = results.fit.t.Nhat;
What_train  = results.fit.t.What;

% Use the HO-SVD decomposition only when reduced dim
Pk = cell(1,ndims(Xtest));
isid = (Nhat_train==size(What_train));
for dimk = 1:ndims(Xtest)
    if Nhat_train(1,dimk)==size(What_train,dimk)
        Pk{1,dimk} = eye(Nhat_train(1,dimk));
    else
        Pk{1,dimk} = Uk{1,dimk};
    end
end
Pk{1,1} = tensor_remove(Pk{1,1},1,idtest);



fit_param_mlr.fit_offset_dim = zeros(1,ndims(What_train));
fit_param_mlr.Pk = Pk;

% Init the core tensors
Wtmp  = 0.001*randn(Nhat_train);
Vtmp = results_mlr.Vhat;
dimcur = Nhat_train;

% Build the full init tensor
for dimk = 1:ndims(What_train)
    dimcur(1,dimk) = size(Xtest_mi,dimk);
    Wtmp = Pk{1,dimk}*tensor_unfold(Wtmp,dimk);
    Wtmp = tensor_fold(Wtmp,dimcur,dimk);
end

% Init lagragians
Dstmp = rand([size(What_train),ndims(What_train)]);
Dstmp = tensor_fold(tensor_unfold(Dstmp,ndims(Dstmp)).*(1-isid)',size(Dstmp), ndims(Dstmp));

% Initialisation
init_param = struct();
init_param.W  = Wtmp;
init_param.V  = tensor_remove(Vtmp, 1,idtest);
init_param.Ds = tensor_remove(Dstmp,1,idtest);

% Launch multilinear rank regression on test dataset
results_mlr2 = tensor_mlr(Xtest_mi,fit_param_mlr,init_param);



tcp_param = results.fit_param.cp;
dim_updt = zeros(1,ndims(Xtest)); dim_updt(1,end) = 1;
tcp_param.dim_updt = dim_updt;

CPtmp = results.fit.CP;
CPtmp{1,1}(idtest,:) = [];
tcp_param.CPfull   = CPtmp;


results_tcp2 = tensor_mlrtcp(results_mlr2, tcp_param);

CP = results_tcp2.fit.CP;
CP{1,1} = results.fit.CP{1,1};
offset = results.fit.offset;

f_link = results.fit_param.t.f_link;

Xtmp = f_link (tensor_reconstruct(CP)+offset);
Xpred = tensor_remove(Xtmp,1,idtrain);





end