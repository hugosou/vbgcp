function X = HadamardProd(U,varargin)
% Customized Hadamard Product
if ~iscell(U), U = [U varargin]; end

X = ones(size(U{1,1},2));
for n=1:size(U,2)
    UtU = U{1,n}'*U{1,n};
    X = X.*UtU;
end
end
