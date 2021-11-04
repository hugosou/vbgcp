function [indicatorff,corespid,coresptype] = get_indicator(record_types,dict_tot)
% Build an indicator function which separates 'neurons' based on record_types
indicator_sizes = zeros(1,1+size(record_types,2));
indicator_sizes(1,1) = size(record_types,1);

for jj=2:length(indicator_sizes)
   indicator_sizes(1,jj) = length(unique(record_types(:,jj-1)));
end

indicator = zeros(indicator_sizes);
offsetm = [1,cumprod(indicator_sizes(1,1:end-1))];




for nn=1:size(record_types,1)
   offs =  -1+indicator_sizes*0; offs(1) = 0;
   tensor_id = [nn,record_types(nn,:)]+offs;
   unitid = tensor_id*offsetm';
   indicator(unitid)=1;  
end


indicatorf  = tensor_unfold(indicator,1);
indicatorff = indicatorf(:,find(sum(indicatorf,1)));


corespid = zeros(size(indicatorff,2),size(record_types,2));
for nr = 1:size(indicatorff,2)
    indicatorfcur = indicatorff(:,nr);
    
    i1 = find(indicatorfcur); i1 = i1(1);
    corespid(nr,:) = record_types(i1,:);
end

if nargin>1
    coresptype = cell(size(corespid));
    for ii = 1:size(corespid,1)
        for jj=1:size(corespid,2)
            Pgt = cellstr(dict_tot{1,jj}(corespid(ii,jj)));
            coresptype{ii,jj} = Pgt{1,1};
        end
    end
end

end


