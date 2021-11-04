function X = tensor_reconstruct(A)
% Reconstruct a tensor from its CP-Decomposition A
% If X is a Tensor of size d1 x d2 x ... x dN
% A must be a cell array of size 1 x N
% A{1,i} is a matrix of size di x R
% R is the number of rank-1 tensor of the CP-Decomposition

R = size(A{1,1},2);

Xdims = zeros(1,size(A,2));
for dimcur = 1:size(A,2)
    Xdims(1,dimcur) = size(A{1,dimcur},1);
end
X = zeros(Xdims);


for rr=1:R
    Ar = A{1,1};
    Xcur = Ar(:,rr);
    
    for dimi=2:length(Xdims)
        Ar = A{1,dimi};
        %Ar
        %rr
        Ar = Ar(:,rr);
        Xcur = tensor_fold(Ar*Xcur(:)',Xdims(1,1:dimi) , dimi);
    end
    
    X = X+Xcur;
end

end
