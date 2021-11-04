function vi_var = vi_update_CP(vi_var,vi_param,Xobs)
% Variational update of the CP factors

% Grasp Current Variational Parameters
CP_mean     = vi_var.CP_mean;
CP_variance = vi_var.CP_variance;
Ulatent = vi_var.latent_mean;
Voffset = vi_var.offset_mean;
Eshape  = vi_var.shape;

% Dimension to be updated
update_CP_dim = vi_param.update_CP_dim;

% Priors
CP_prior_mean      = vi_var.CP_prior_mean;
CP_prior_precision = vi_var.CP_prior_precision;

% Problem size
Xdims = size(Xobs);

% Tensor Rank
R = size(CP_mean{1},2);

% Diagonal Elements for Precision
RR = (0:(R-1))*(R+1)+1;

% Deal with missing data
observed_data = vi_param.observed_data;

% Deal with sparse data
if strcmp(vi_param.sparse,'block')
    observed_data_block   = vi_var.observed_data_block;
    observed_data_bloc_id = vi_var.observed_data_bloc_id;
end

if strcmp(vi_param.sparse,'block')
    
    % Exploit Block Structure of observed_data
    for dimn = 1:length(Xdims)
        if update_CP_dim(dimn)
            % Loop on dimensions: n-th Unfoldings
            
            Z  = ((Xobs-Eshape)/2 - Voffset.*Ulatent).*observed_data;
            Zn = tensor_unfold(Z,dimn);
            Un = tensor_unfold(Ulatent,dimn);
            Bn  = KhatriRaoProd_mean(CP_mean,dimn);
            BBn = KhatriRaoProd_var(CP_mean,CP_variance,dimn);
            
            % Grasp Observed Blocks
            On_block_cur = observed_data_block{1,dimn};
            In_block_cur = observed_data_bloc_id{1,dimn};
            
            % Block Priors
            prec_prior = CP_prior_precision{1,dimn};
            mean_prior = CP_prior_mean{1,dimn};
            
            for blockm = 1:length(On_block_cur)
                % Loop On current block
                
                % Efficiently deal with missing data
                Unm = Un(In_block_cur{1,blockm},On_block_cur{1,blockm});
                Znm = Zn(In_block_cur{1,blockm},On_block_cur{1,blockm});
                
                % <B'UB>
                BUBm = Unm * BBn(On_block_cur{1,blockm},:);
                
                % <B'><U><Z>
                BUm  = Znm *  Bn(On_block_cur{1,blockm},:);
                
                % Current Priors
                prec_prior_m = prec_prior(In_block_cur{1,blockm},:);
                mean_prior_m = mean_prior(In_block_cur{1,blockm},:);
                
                % Temporary Update (precisions are diag)
                prec_post_m = prec_prior_m + BUBm;
                mean_post_m_tmp = BUm + prec_prior_m(:,RR).*mean_prior_m;
                
                % Invert Precision and Update mean
                mean_post_m = zeros(size(mean_post_m_tmp));
                vari_post_m = zeros(size(prec_post_m));
                
                for diminm=1:size(BUm,1)
                    vari_post_mi = inv(reshape(prec_post_m(diminm,:),[R,R]));
                    mean_post_mi = vari_post_mi*mean_post_m_tmp(diminm,:)';
                    vari_post_mi = vari_post_mi(:)';
                    
                    mean_post_m(diminm,:) = mean_post_mi;
                    vari_post_m(diminm,:) = vari_post_mi;
                end
                
                % Store
                CP_mean{1,dimn}(In_block_cur{1,blockm},:) = mean_post_m;
                CP_variance{1,dimn}(In_block_cur{1,blockm},:) = vari_post_m;
            end
        end
        
        
    end
    
else
    % Full observed_data or few missing
    for dimn = 1:length(Xdims)
        if update_CP_dim(dimn)
            % Loop on dimensions: n-th Unfoldings
            
            Z  = ((Xobs-Eshape)/2 - Voffset.*Ulatent).*observed_data;
            Zn = tensor_unfold(Z,dimn);
            Un = tensor_unfold(Ulatent,dimn);
            Bn  = KhatriRaoProd_mean(CP_mean,dimn);
            BBn = KhatriRaoProd_var(CP_mean,CP_variance,dimn);
            
            % Priors
            prec_prior = CP_prior_precision{1,dimn};
            mean_prior = CP_prior_mean{1,dimn};
            
            % <B'UB>
            BUB = Un*BBn;
            
             % <B'><U><Z>
            BUZ = Zn*Bn;
            
            % Temporary Update (precisions are diag)
            prec_post = prec_prior+BUB;
            mean_post_tmp = BUZ + prec_prior(:,RR).*mean_prior;
            
            % Invert Precision and Update mean
            for dimi = 1:size(BUZ,1)
                vari_post_i = inv(reshape(prec_post(dimi,:),[R,R]));
                mean_post_i = vari_post_i*mean_post_tmp(dimi,:)';
                vari_post_i = vari_post_i(:)';
                
                CP_mean{1,dimn}(dimi,:) = mean_post_i;
                CP_variance{1,dimn}(dimi,:) = vari_post_i;
                
            end
        end
    end
    
end


% Use last unfolding to get <tensor> and <tensor^2>
tensor_mean  = tensor_fold(CP_mean{1,dimn}*Bn',Xdims,dimn);
tensor_2_mean_tmp = kron(ones(1,R),CP_mean{1,dimn}).*repelem(CP_mean{1,dimn}, 1,R);
tensor2_mean = tensor_fold((tensor_2_mean_tmp+CP_variance{1,dimn})*BBn',Xdims,dimn);

% Save Posterior
vi_var.CP_mean = CP_mean;
vi_var.CP_variance = CP_variance;

% Save tensor moments
vi_var.tensor_mean  = tensor_mean;
vi_var.tensor2_mean = tensor2_mean;


end
