function [Anormed,Snormed] = normalize_cp(A,normdim,S,dosort)
% - Normalize a CP tensor decomposition.
%   If X = [|A_1, A_2, ...A_d|], normalise the colums of A_i
%   Gather the scaling coefficient in Anormed_(d+1)
% - Inject back the normalizer in dimension normdim
% - Normalize posterior variance S if provided

% Sort CP by amplitude
if nargin <4
    dosort = 1;
end

% Deal with CP posterior probabilities
normalize_variance = 1;
if (nargin <3) || isempty(S)
   normalize_variance = 0; 
end
 
if nargin <2
    normdim = size(A,2)+1;
end

% Init CP component and variances
Anormed = cell(1,size(A,2)+1);
Snormed = cell(1,size(A,2)+1);

% Tensor Rank
R = size(A{1,1},2); lambdas = ones(1,R);

for dimcur = 1:size(A,2)
    
    % Norm of the current component
    normi = sqrt(diag(A{1,dimcur}'*A{1,dimcur}))';
    
    % Normalize Means
    Anormed{1,dimcur} = A{1,dimcur}./(eps+normi);
    
    % Save Normalizer
    lambdas = lambdas.*normi;
    
    % Convention for CP factors signs
    signeor = sign(sum(Anormed{1,dimcur}));
    signeor(find(signeor==0)) = 1;
    
    Anormed{1,dimcur} = Anormed{1,dimcur} .* signeor;
    lambdas = lambdas.*signeor;
    
    % Normalize Variances
    if normalize_variance
        Snormed{1,dimcur} = S{1,dimcur}./...
            reshape(normi(:)*normi(:)'+eps,1,R*R);
    end
    
end

% Scaler defined  positive. Inject sign in last dim if not. 
Anormed{1,dimcur} = Anormed{1,dimcur}.*sign(lambdas);
lambdas = lambdas.*sign(lambdas);

% Augment factors with normalizer
Anormed{1,size(A,2)+1} = lambdas;
Snormed{1,size(A,2)+1} = zeros(1,R*R);

% Sort PC
if dosort
    
    [~,sorted_id_mean] = sort(abs(lambdas),'descend');
    sorted_id_var = reshape(1:(R*R),R,R);
    sorted_id_var = permute_variance(sorted_id_var,sorted_id_mean);
    
    for dimcur = 1:(size(A,2)+1)
        Anormed{1,dimcur} = Anormed{1,dimcur}(:,sorted_id_mean);
        
        if normalize_variance
            Snormed{1,dimcur} = Snormed{1,dimcur}(:,sorted_id_var(:));
        end
    end
end

% Inject back normalizer in normdim
if normdim <=size(A,2)
    if normalize_variance
        [Anormed,Snormed] = absorb_normalizer(Anormed,normdim,Snormed);
    else
        Anormed = absorb_normalizer(Anormed,normdim);
    end

end

end

function Sp = permute_variance(S,permid)

Sp = zeros(size(S));
for ii=1:size(S,1)
    for jj=1:size(S,1)
        Sp(ii,jj) = S(permid(ii),permid(jj));
    end
end
end