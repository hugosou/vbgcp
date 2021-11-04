function gathered_models = gather_cps(models)
% Gather CP decomposition by model paramameters and factor id
% (must have reorder CP first)

mdims = ndims(models);
msize = size(models);

Ntest = msize(end);
Ngrid = prod(msize(1:end-1));

% Use linear index to use flexible sizes for models
linear_index_tot = reshape(1:prod(msize),msize);
linear_index_tot = permute(linear_index_tot, [mdims,1:mdims-1]);
linear_index_tot = reshape(linear_index_tot,[Ntest,Ngrid]);


if length(msize(1:end-1))==1
    gathered_models = cell(msize(1:end-1),1);
else
    gathered_models = cell(msize(1:end-1));
end

for ngrid=1:Ngrid
    dimI = size(models{linear_index_tot(1,ngrid)},2);
    
    if not(isempty(models{linear_index_tot(1,ngrid)}))
        R = size(models{linear_index_tot(1,ngrid)}{1,1},2);
        gathered_models{ngrid} = cell(dimI,R);
        for dimi = 1:dimI
            for r= 1:R
                Di = size(models{linear_index_tot(1,ngrid)}{1,dimi},1);
                CPir = zeros(Di,Ntest);
                for ntest = 1:Ntest
                    
                    CPir(:,ntest) = models{linear_index_tot(ntest,ngrid)}{1,dimi}(:,r);
                end
                gathered_models{ngrid}{dimi,r} = CPir;
            end
        end
    end
end
end


