function [smlty,perm_final,sign_final] = get_similarity(models, ref_model,used_dims)
% Estimate the similarity index from a cell array of CP-Tensor models of
% same rank using the permutations contained in permutations
% Metric from (Tomasi & Bros 2004)

% Model of reference (usually the best one)
ref_model = models{1,ref_model};

Nmodel   = size(models,2);
TensRank = size(ref_model{1,1},2);
TensDims = size(ref_model,2);

if nargin<3
   used_dims = ones(1,TensDims);
end


if not(TensDims==size(models{1,1},2))
    error('Invalid Dimmensions. Models should all be non normalized.')
end

% Normalize reference model
ref_model  = normalize_cp(ref_model);
ref_lambda = ref_model{1,TensDims+1};


smlty = zeros(1,Nmodel);

sign_final = zeros(Nmodel , TensRank, TensDims); 
perm_final = zeros(Nmodel,TensRank);

for nmodel=1:Nmodel
    
    % Normalize current model
    cur_model = normalize_cp(models{1,nmodel});
    
    % Calculate a_r a_p(r) o b_r b_p(r) o ... for all permutations
    cur_product = ones(TensRank,TensRank);
    
    signtot = cell(1,TensDims);
    for dimcur=1:TensDims
        vref = ref_model{1,dimcur};
        vcur = cur_model{1,dimcur};
        
        signtot{1,dimcur} = sign(vref'*vcur)';
        if used_dims(1,dimcur)
            cur_product = cur_product.*(vref'*vcur);
        else 
            cur_product = cur_product.*sign(vref'*vcur);
        end
           
    end
    
    % Calculate (1 - |lambda_r-lambda_r_p(r)|/max(lambda_r,lambda_p(r))) / R
    cur_lambda = cur_model{1,TensDims+1};
    dif12 = abs(cur_lambda-ref_lambda');
    max12 =(abs(cur_lambda+ref_lambda')+abs(cur_lambda-ref_lambda'))/2;  

    
    % Get the full similarities for ALL permutations
    Rapp = sum(ref_model{end}>eps); % In case some CP are zeros
    cur_product = (1-dif12./(max12+eps)).*cur_product/Rapp;
    
    
    % Use Munkres (Hungarian) Algorithm for Linear Assignment Problem on cur_product
    offn = (0:1:(size(cur_product,1)-1))*size(cur_product,1);
    cur_productt = cur_product';
    
    
    if length(find(isnan(cur_product)))>1
        cur_perm = 1:size(cur_product,1);
        cur_sum  = NaN;
        warning('isNan Values')
    else
        cur_perm = munkres(-cur_product); 
        cur_sum  = sum(cur_productt(offn+cur_perm));
        %cur_sig  =  signtot(offn+cur_perm);  
        cur_sig = cellfun(@(x,y) x(y), signtot,num2cell(repelem((offn+cur_perm),TensDims,1),2)' , 'UniformOutput',false);

    end
    
    cur_sig = cell2mat(cur_sig')'; 
    cur_sig(find(cur_sig==0)) = 1;
    
    % Store similarity index and corresponding permutation
    smlty(1,nmodel)      = cur_sum;
    perm_final(nmodel,:) = cur_perm;
    sign_final(nmodel,:,:) = cur_sig;
    smlty(find(isnan(smlty))) = 0;
    
end



end
