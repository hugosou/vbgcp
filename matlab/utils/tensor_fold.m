function Y = tensor_fold(YYi,DIMS,DIMi)
% Returns full tensor Y of size DIMS from its (DIMi)th unfolding YYi

if not(DIMS(DIMi)==size(YYi,1))
    error('Invalid Dimensions')
end

neqdim = DIMS;
neqdim(DIMi) = [];

Ytmp= reshape(YYi, [DIMS(DIMi),neqdim ]);

neqdim_id = 1:length(DIMS);
neqdim_id(DIMi)=[];

dim_permutation = zeros(1,length(DIMS));
dim_permutation(DIMi) = 1;
dim_permutation(neqdim_id) = 2:length(DIMS);

Y = permute(Ytmp,dim_permutation);
end
