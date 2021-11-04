function KRPn = KhatriRaoProd_mean(factors,dimn)
% INPUTS : factors = {A_1, ..., A_D}.
% OUTPUTS: KRPn = A_D * ... * A_n+1 * A_n-1 * ... * A_1
%          '*': Khatri-Rao product

Xdims = cellfun(@(Z) size(Z,1) ,factors);

% Grasp and flip other dimensions
not_dimn = 1:length(Xdims);
not_dimn(dimn)=[];
not_dimn = fliplr(not_dimn);

% Khatri-Rao Product
KRPn = KhatriRaoProd(factors{1,not_dimn});

end