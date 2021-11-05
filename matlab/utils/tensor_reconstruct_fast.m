function X = tensor_reconstruct_fast(factors, core)
% Reconstruct a tensor from a Tucker or a CP decomposition.
% If no core tensor is provided, a rank R CP decomposition is assumed.

DIMS = cellfun(@(CP) size(CP,1), factors);
dIMS = cellfun(@(CP) size(CP,2), factors);
NDIMS = length(DIMS);


if nargin <2
    R = unique(dIMS);
    if length(R)>1
        error('If core not provided, factors must correspond to CP')
    end
    
    % The core correspond to the R x ... x R identity
    core =  zeros(R*ones(1,NDIMS));
    core(eyeid(R,NDIMS)) = 1;
end

X = xtensor(cat(2,core,factors), cat(2,-(1:NDIMS),num2cell([1:NDIMS;-(1:NDIMS)]',2)'));

function n = eyeid(R,NDIMS)
    y = (1:R)'*ones(1,NDIMS);
    n = 1 + [1,R.^(1:(NDIMS-1))]*(y-1)';
end

end
