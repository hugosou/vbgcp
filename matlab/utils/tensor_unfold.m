function YYi = tensor_unfold(Y, DIMi)
% Returns (DIMi)th unfolding of tensor Y

dims = size(Y);

if all(size(Y)==1)
    YYi = Y;
else
    if DIMi>length(dims)+1
        error('Requested unfolding dimension bigger than tensor dimension')
    elseif DIMi==length(dims)+1
        YYi = Y(:)';
    elseif DIMi<=length(dims)
        
        
        neqdim = 1:length(dims);
        neqdim(DIMi) = [];
        
        dim_permutation = [DIMi, neqdim];
        
        new_dim_1 = dims(DIMi);
        new_dim_2 = prod(dims(neqdim));
        
        YYi = reshape(permute(Y, dim_permutation),[new_dim_1,new_dim_2]);
        
    end
end
end
