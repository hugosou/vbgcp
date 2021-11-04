function Xpred = lno_test_gcp(results,idtest,idtrain, Xtest)

Xtest_mi = tensor_remove(Xtest,1,idtest);

fit_param = results.fit_param;

fit_param.fit_decomp_dim = zeros(1,ndims(Xtest));
fit_param.fit_decomp_dim(1,ndims(Xtest)) = 1;
fit_param.fit_offset_dim = zeros(1,ndims(Xtest));

% Grasp trained factors and offset
trained_factors = results.fit.CP;
trained_factors{1,1}(idtest,:) = [];
trained_offset = results.fit.offset;

% Build init structure to fix offset and trained factors
fit_init = struct();
fit_init.moments = init_moments(size(Xtest_mi),results.fit_param.R);
fit_init.offsets = tensor_remove(trained_offset,1,idtest);
fit_init.factors = trained_factors;

% Launch fit
results2 = tensor_gcandelinc(Xtest_mi,fit_param,fit_init);
test_factors = results.fit.CP;
test_factors{1,4} = results2.fit.CP{1,4};
f_link = results.fit_param.f_link;

Xtmp = f_link (tensor_reconstruct(get_constrained_factor(test_factors, results.fit_param))+trained_offset);
Xpred = tensor_remove(Xtmp,1,idtrain);

end




