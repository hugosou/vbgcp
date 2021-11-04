function [mnew,Snew] = squeeze_missing(m,S,observed_data,dimdisc,an)
% Use observed_data matrix to squeeze CP dimensions in dimdisc
% Produces an estimate of the mean m and variance S using moment matching

if nargin <5
    an = 3;
end

squeeze_variance = 1;
if isempty(S)
    squeeze_variance = 0;
end

dimkeep = 1:length(m); 
dimkeep(dimdisc) = [];

% Keep only the relevant dimensions for observed data
if ndims(observed_data)>2
    odims = size(observed_data);
    observed_data = permute(observed_data, [dimdisc,dimkeep]);
    observed_data = reshape(observed_data, [prod(odims(dimdisc)),prod(odims(dimkeep))]);
    observed_data = observed_data(:,1);
    observed_data = reshape(observed_data, odims(dimdisc));
end


[m,S] = normalize_cp(m,an,S);

% Squeeze means
mnew = cell(1,size(m,2)-1);
mnew(2:(size(m,2)-1)) = m(dimkeep);



d1 = size(m{dimdisc(1)},1);
d2 = size(m{dimdisc(2)},1);

if all([d1,d2] == size(observed_data'))
    observed_data=observed_data';
end 
assert(all([d1,d2] == size(observed_data)))

% Squeeze variances
if squeeze_variance
    Snew = cell(1,size(S,2)-1);
    Snew(2:(size(S,2)-1)) = S(dimkeep);
else
    Snew = {};
end



for dimi=1:d1
    m1 = m{dimdisc(1)}(dimi,:);
    m2 = sum(observed_data(dimi,:)'.*m{dimdisc(2)},1);
    
    mnewi = m1.*m2;
    mm1 = m1(:)*m1(:)';
    mm2 = m2(:).*m2(:)';
    mnew{1}(dimi,:) = mnewi;
    
    % Moment Matching of the variable product
    if squeeze_variance
        S1 = S{dimdisc(1)}(dimi,:);
        S2 = sum(observed_data(dimi,:)'.*S{dimdisc(2)},1);
        Snewi = S1.*S2 + mm1(:)'.*S2 + S1.*mm2(:)';
        Snew{1}(dimi,:) = Snewi;
    end
    
end

end
