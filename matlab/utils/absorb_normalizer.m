function [A,S] = absorb_normalizer(Anormed,normdim,Snormed)
% Inject Back Normalizer of CP decomposition: X = [|A_1, A_2, ...A_d|]

assert(size(Anormed,2)>1)
dimmax = size(Anormed,2)-1;

% Normalize is in last dimension
normalizer = Anormed{1,dimmax+1};

% Factors
A = cell(1,dimmax);
A(1:dimmax) = Anormed(1:dimmax);
A{1,normdim} = A{1,normdim}.*normalizer;

% Factors Variance (if any)
if nargin >2
S = cell(1,dimmax);
    var_normalizer = reshape(normalizer(:)*normalizer(:)',1,length(normalizer)^2);
    S(1:dimmax) = Snormed(1:dimmax);
    S{1,normdim} = S{1,normdim}.*var_normalizer;
end

end