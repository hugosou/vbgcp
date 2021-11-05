function models_squeezed = squeeze_model(models, observed_data, dimoi, an)
% Squeeze model according to observed_data block structure

assert(sum(dimoi)==2,'Incorrect Dimensions')

dimdisc = find(dimoi);
dimkeep = 1:length(dimoi); dimkeep(dimdisc) = [];

models_squeezed = cell(size(models));

if nargin<4
    an = 3;
end

% Keep ponly the relevant dimensions for observed data
if ndims(observed_data)>2
    odims = size(observed_data);
    observed_data = permute(observed_data, [dimdisc,dimkeep]);
    observed_data = reshape(observed_data, [prod(odims(dimdisc)),prod(odims(dimkeep))]);
    observed_data = observed_data(:,1);
    observed_data = reshape(observed_data, odims(dimdisc));
end

assert(all(sum(observed_data,2)==1),'Incorrect Observed Data')


for model_id = 1:numel(models)
    if not(isempty(models{model_id}))
        
        model_cur = absorb_normalizer(normalize_cp(models{model_id}),an);
        
        model_new = cell(1,size(model_cur,2)-1);
        model_new(2:(size(model_cur,2)-1)) = model_cur(dimkeep);
        
        for rr = 1:size(model_cur{1},2)
            model_new{1}(:,rr) = sum(model_cur{dimdisc(1)}(:,rr).*observed_data.*model_cur{dimdisc(2)}(:,rr)',2);
        end
        
        models_squeezed{model_id} = model_new;
        
    end
end

end