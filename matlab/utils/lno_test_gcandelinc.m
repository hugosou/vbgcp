function Xpred = lno_test_gcandelinc(results,idtest,idtrain, Xtest)
    Xpred = tensor_remove(results.fit.Xhat,1,idtest);
end