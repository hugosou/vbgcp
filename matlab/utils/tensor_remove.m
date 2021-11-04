function Xr = tensor_remove(X,dimi,id)
    %  Use i-th unfolding X_(dimi) of tensor X and remove id-th lines
    Xdim = size(X);
    Xdim(1,dimi) = Xdim(1,dimi)-length(id);
    Xi = tensor_unfold(X,dimi);
    Xi(id,:) = [];
    Xr = tensor_fold(Xi,Xdim,dimi); 
end