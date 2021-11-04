function [smlty,perm_final,sign_final] = get_similarity_missing(models, ref_model,observed_data,combined_dims)
% Similarity metric accross models with missing data

if nargin <3
    % if no missing data, use standard metric
    [smlty,perm_final,sign_final] = get_similarity(models, ref_model);
    
else
    assert(length(find(combined_dims)) == 2, 'Invalid Number of Combined Dimensions. Must be 2.')
    
    
    % Model of reference (usually the best one)
    ref_model = models{1,ref_model};
    
    % Number of models
    num_model   = size(models,2);
    
    % Order of the decomposition
    tensor_rank = size(ref_model{1,1},2);
    tensor_dims = size(ref_model,2);
    
    %  Dimensions that needs to be combined or not
    cmbd_dims = find(combined_dims);
    stdr_dims = 1:tensor_dims; stdr_dims(find(combined_dims)) = [];
    
    % Get the relevant matrix of oberved data
    obsdims = size(observed_data);
    kptdims = find(combined_dims);
    dscdims = 1:length(obsdims); dscdims(kptdims)=[];
    observed_data_oi = reshape(permute(observed_data, [kptdims,dscdims]), [obsdims(kptdims), prod(obsdims(dscdims))]);
    observed_data_oi = observed_data_oi(:,:,1);
    
    % Combine factors and extract observed ones
    combine_observed_factors = @(X,Y) cellfun(@(x,y) (x*y').*observed_data_oi,...
        num2cell(X,1),num2cell(Y,1), 'UniformOutput',false);
    
    % Angle between 2 matrices
    matrix_angle = @(X,Y) (X(:)'*Y(:))/sqrt((X(:)'*X(:))*(Y(:)'*Y(:)));
    
    % Normalize reference model
    ref_model  = normalize_cp(ref_model);
    ref_lambda = ref_model{1,tensor_dims+1};
    
    smlty = zeros(1,num_model);
    sign_final = zeros(num_model , tensor_rank, tensor_dims);
    perm_final = zeros(num_model,size(models{1,1}{1,1},2));
    
    % Pairwise similarities
    for nmodel=1:num_model
        
        % Normalize current model
        cur_model = normalize_cp(models{1,nmodel});
        
        % Calculate a_r a_p(r) o b_r b_p(r) o ... for all permutations
        cur_product = ones(tensor_rank,tensor_rank);
        
        
        signtot = cell(1,tensor_dims);
        % Non Combined-Dimmensions
        for dimcur = stdr_dims
            vref = ref_model{1,dimcur};
            vcur = cur_model{1,dimcur};
            signtot{1,dimcur} = sign(vref'*vcur)';
            
            cur_product = cur_product.*(vref'*vcur);
        end
        
        % Angle between all possible combination of vref and vcur
        combine_factors_angle = cellfun(@(X,Y) matrix_angle(X,Y),...
            repmat(combine_observed_factors(ref_model{1,kptdims(1)},ref_model{1,kptdims(2)})',[1,size(ref_model{1},2)]),...
            repmat(combine_observed_factors(cur_model{1,kptdims(1)},cur_model{1,kptdims(2)}) ,[size(cur_model{1},2),1]));
       
        cur_product = cur_product.*combine_factors_angle;
        signtot{1,cmbd_dims(1)} = sign(combine_factors_angle);
        signtot{1,cmbd_dims(2)} = ones(size(combine_factors_angle));        
 
        % Calculate (1 - |lambda_r-lambda_r_p(r)|/max(lambda_r,lambda_p(r))) / R
        cur_lambda = cur_model{1,tensor_dims+1};
        dif12 = abs(cur_lambda-ref_lambda');
        max12 =(abs(cur_lambda+ref_lambda')+abs(cur_lambda-ref_lambda'))/2;
        

        % Get the full similarities for ALL permutations
        cur_product = (1-dif12./max12).*cur_product/tensor_rank;
        
        
        % Use Munkres (Hungarian) Algorithm for Linear Assignment Problem on cur_product
        offn = (0:1:(size(cur_product,1)-1))*size(cur_product,1);
        cur_productt = cur_product';
        
        
        if length(find(isnan(cur_product)))>1
            cur_perm = 1:size(cur_product,1);
            cur_sum  = NaN;
            warning('isNan Values')
        else
            cur_perm = munkres(-(cur_product));
            
            cur_sum  = sum(cur_productt(offn+cur_perm));
            
            %cur_sig  =  signtot(offn+cur_perm);
            
            cur_sig = cellfun(@(x,y) x(y), signtot,num2cell(repelem((offn+cur_perm),tensor_dims,1),2)' , 'UniformOutput',false);
            
            
        end
        
        
        % Store similarity index and corresponding permutation
        smlty(1,nmodel)      = cur_sum;
        perm_final(nmodel,:) = cur_perm;
        sign_final(nmodel,:,:) = cell2mat(cur_sig')';
        
    end
    
    
    
    
    
    
    
    
    
    
    
    
end



end