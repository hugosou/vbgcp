function [models_reordered,variances_reordered] = reorder_cps(models,permt_tot,sign_tot,variances)
% REORDER_CPS uses optimal permutations discovered with similarity measures
% to reorder (and normalize) CP factors and variance if provided.


if nargin <4
    reorder_variances = 0;
else
    reorder_variances = 1;
    variances_reordered = cell(size(models));
end

models_reordered = cell(size(models));


mdims = ndims(models);
msize = size(models);

Ntest = msize(end);
Ngrid = prod(msize(1:end-1));

% Use linear index to use flexible sizes for models
linear_index_tot = reshape(1:prod(msize),msize);
linear_index_tot = permute(linear_index_tot, [mdims,1:mdims-1]);
linear_index_tot = reshape(linear_index_tot,[Ntest,Ngrid]);


for ngrid=1:Ngrid
    % Gather models with common parameters
    perms_cur = permt_tot{ngrid};
    
    if not(isempty(perms_cur))
        
        if nargin<3
            signe_cur = ones(Ntest,1,size(models{1},2));
        else
            signe_cur = sign_tot{ngrid};
        end
        
        for ntest = 1:Ntest
            % Current id
            linear_index_cur = linear_index_tot(ntest,ngrid);
            
            % Best Perumtations
            permt_cur = perms_cur(ntest,:);
            
            
            if not(reorder_variances) || isempty(variances{linear_index_cur})
                
                
                model_cur = normalize_cp(models{linear_index_cur},3);
                model_new = cell(size(model_cur));
                
                % Permute factors
                for dimi = 1:size(model_cur,2)
                    model_new{1,dimi} = model_cur{1,dimi}(:,permt_cur) .* signe_cur(ntest,:,dimi);
                    
                    
                end
                
                models_reordered{linear_index_cur} = model_new;
                
                
            else 
                
                model_cur = models{linear_index_cur};
                variance_cur = variances{linear_index_cur};
                
                [model_cur , variance_cur] = normalize_cp(model_cur,3,variance_cur);
                
                model_new = cell(size(model_cur));
                variance_new = cell(size(model_cur));
                
                R = size(model_cur{1},2);
                perm_cur_var = reshape(1:(R*R),R,R);
                perm_cur_var = permute_variance(perm_cur_var,permt_cur);
                
                % Permute factors
                for dimi = 1:size(model_cur,2)
                    model_new{1,dimi} = ...
                        model_cur{1,dimi}(:,permt_cur) .*signe_cur(ntest,:,dimi);
                    
                    variance_new{1,dimi} =...
                        variance_cur{1,dimi}(:,perm_cur_var(:));
                end
 
                models_reordered{linear_index_cur} = model_new;
                variances_reordered{linear_index_cur} = variance_new;
                
            end
            
        end
        
    end
    
end


end



function Sp = permute_variance(S,permid)

Sp = zeros(size(S));
for ii=1:size(S,1)
    for jj=1:size(S,1)
        Sp(ii,jj) = S(permid(ii),permid(jj));
    end
end
end

