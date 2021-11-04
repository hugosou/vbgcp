function [test_folder,train_folder] = xval_idx(ids,k_test,kmax)
% Indices for cross validation
k_tot = length(ids);
test_folder  = nchoosek(ids,k_test);
ntot = nchoosek(k_tot,k_test);


train_folder = zeros(size(test_folder,1), k_tot-k_test);

for ll=1:size(train_folder,1)
    train_folder(ll,:)=ids(find(not(ismember(ids, test_folder(ll,:)))));
end

if nargin>2
    
    if(kmax < ntot)
        
        pp = randperm(ntot);
        pp = pp(1:kmax);
        
        test_folder  = test_folder(pp,:);
        train_folder = train_folder(pp,:);
        
    end
end

end