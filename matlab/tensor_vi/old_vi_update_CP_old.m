function vi_var = vi_update_CP(vi_var,vi_param,Xobs)
% Variational update of the CP factors

% Grasp Current Variational Parameters
CP_mean     = vi_var.CP_mean;
CP_variance = vi_var.CP_variance;
Ulatent = vi_var.latent_mean;
Voffset = vi_var.offset_mean;
Eshape  = vi_var.shape;

% Dimension to be update
update_CP_dim = vi_param.update_CP_dim;

% Priors
CP_prior_mean      = vi_var.CP_prior_mean;
CP_prior_precision = vi_var.CP_prior_precision;

% Problem size
Xdims = size(Xobs);

% Tensor Rank
R = size(CP_mean{1},2);

% Deal with missing data
Oni_tot = vi_var.observed_ni;

for dimn = 1:length(Xdims)
    if update_CP_dim(dimn)
        % Loop on dimensions: n-th Unfoldings
        
        Xn = tensor_unfold(Xobs,dimn);
        Un = tensor_unfold(Ulatent,dimn);
        Vn = tensor_unfold(Voffset,dimn);
        On = Oni_tot{dimn};
        Zn = (Xn - Eshape)./(2*Un+eps) - Vn;
        Bn  = KhatriRaoProd_mean(CP_mean,dimn);
        BBn = KhatriRaoProd_var( CP_mean,CP_variance,dimn);
        
        for dimi=1:Xdims(dimn)
            % Loop on dimensions: i-th row
            
            % We only keep index corresponding to observed data
            Oni = On{dimi};
            
            % <Z> and <U> with only the relevant index
            Zni = Zn(dimi,Oni);
            Uni = Un(dimi,Oni);
            
            % <B>'<U>
            BntUni  = Bn(Oni,:)'.*Uni;
            
            % <B'diag(U) B>
            %BntUniBni = reshape(sum(BBn(Oni,:)'.*Uni,2),R,R);
            BntUniBni = reshape(Uni*BBn(Oni,:),[R,R]);
            
            % Prior Covariance and Mean
            Lni  = reshape(CP_prior_precision{1,dimn}(dimi,:),[R,R]);
            Mni  = CP_prior_mean{1,dimn}(dimi,:)';
            
            % Posterior Mean/Variance
            Var_ni = inv(BntUniBni + Lni);
            Mea_ni = Var_ni * (BntUni * Zni' +  Lni*Mni );
            
            % Save Mean/Variance
            CP_mean{1,dimn}(dimi,:) = Mea_ni;
            CP_variance{1,dimn}(dimi,:) = Var_ni(:);
            
        end
    end
end

% Get reconstructed tensor 1st and second moment
AAt = get_AAt(CP_mean,CP_variance);
%tensor_mean  = tensor_reconstruct_fast(CP_mean);
%tensor2_mean = tensor_reconstruct_fast(AAt);
tensor_mean  = tensor_reconstruct(CP_mean);
tensor2_mean = tensor_reconstruct(AAt);


% Save Posterior
vi_var.CP_mean = CP_mean;
vi_var.CP_variance = CP_variance;

% Save tensor moments
vi_var.tensor_mean  = tensor_mean;
vi_var.tensor2_mean = tensor2_mean;

end
