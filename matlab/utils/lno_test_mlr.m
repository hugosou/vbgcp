function Xpred = lno_test_mlr(results,idtest,idtrain, Xtest)
Xtest_mi = tensor_remove(Xtest,1,idtest);

% Grasp MLR train fit
Nhat_train  = results.fit.Nhat;
What_train  = results.fit.What;
[~,Uk] = tensor_hosvd(What_train,Nhat_train);

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


fit_param = results.fit_param;
fit_param.fit_offset_dim = zeros(1,ndims(What_train));
fit_param.Pk = Pk;

% Init the core tensors
Wtmp  = 0.001*randn(Nhat_train);
Vtmp = results.fit.Vhat;
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

% Launch test fit
results2 = tensor_mlr(Xtest_mi,fit_param,init_param);

% Grasp dynamics tensor and project it to idtrain
What2 = results2.fit.What;
Xpredinv  = tensor_fold(Uk{1,1}*Pk{1,1}'*tensor_unfold(What2,1),size(Xtest),1)+results.fit.Vhat;
Xpred_tot  = results.fit_param.f_link(Xpredinv);
Xpred = tensor_remove(Xpred_tot,1,idtrain);

end