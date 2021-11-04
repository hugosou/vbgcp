function [dk,Dk] = get_dk(Pk)
dk = zeros(1,size(Pk,2));
Dk = zeros(1,size(Pk,2));

for k =1:size(Pk,2)
    dk(1,k) = size(Pk{1,k},2);
    Dk(1,k) = size(Pk{1,k},1);
end

end