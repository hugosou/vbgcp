function Xpredi = lno_wrapper(results,method,idtrain,idtest,Xtest)

if contains(method,'MLR')
    Xpredi = lno_test_mlr(results,idtest,idtrain, Xtest);
    
elseif contains(method,'TCP')
    Xpredi = lno_test_tcp(results,idtest,idtrain, Xtest);
    
elseif contains(method,'TGCPobs')
    Xpredi = lno_test_gcp(results.gcp,idtest,idtrain, Xtest);
    
elseif contains(method,'TGCPhat')
    Xpredi = lno_test_gcp(results.gcp,idtest,idtrain, Xtest);
    
elseif contains(method,'GCP')
    Xpredi = lno_test_mrp(results,idtest,idtrain, Xtest);
    
elseif contains(method,'GCPx2')
    Xpredi = lno_test_mrp(results,idtest,idtrain, Xtest);
    
elseif contains(method,'MRP')
    Xpredi = lno_test_mrp(results,idtest,idtrain, Xtest);
    
elseif contains(method,'GCANDELINC')
    Xpredi = lno_test_gcandelinc(results,idtrain, idtest, Xtest);
end

end