function D = deviance_poisson_nwise(Xobs, Xhat)
N = size(Xobs,1);
D = zeros(1,N);

Xobs1 = tensor_unfold(Xobs,1);
Xhat1 = tensor_unfold(Xhat,1);

for n=1:N
    D(1,n) = deviance_poisson(Xobs1(n,:), Xhat1(n,:));
end

end



