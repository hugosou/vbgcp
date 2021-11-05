function vi_var = vi_update_shape(vi_var,vi_param, Xobs)
% Update model shape parameter

% Update method
shape_update  = vi_param.shape_update;

% Deal with missing entries
observed_data = vi_param.observed_data;

if all(observed_data(:)==1)
    % All data are observed
    observed_id = (1:numel(Xobs))';
else
    % There is missing data.
    observed_id = find(observed_data);
end

if strcmp(shape_update,'MM-G')
    % Moment Match a Gamma distributions from which we estimate KL
    
    % Restrick Dataset for fast shape update
    %Nlimit = 25000;
    Nlimit = length(observed_id);
    observed_id = observed_id(randperm(length(observed_id),Nlimit));
    
    % Observed data
    Xobsi = Xobs(observed_id(:));
    
    % <W> and <W^2>
    Wm  = vi_var.tensor_mean(observed_id(:));
    Wm2 = vi_var.tensor2_mean(observed_id(:));
    
    % <V> and <V^2>
    Vm  = vi_var.offset_mean(observed_id(:));
    Vm2 = Vm.^2 + vi_var.offset_variance(observed_id(:));
    
    % q(latent) ~ PG(X+shape_old,Om)
    shape_old = vi_var.shape;
    Om = sqrt(Vm2 + Wm2 + 2*Wm.*Vm);
    
    % Moment match PG(1, Om) with Gamma(alpha0,beta0)
    moments = pg_moment(ones(length(Om(:)),1),Om(:),1);
    alpha_0 = moments(:,1).^2./moments(:,2);
    
    % Use linearity in shape and Closed Form KL for gamma to estimate FE(shape)
    FE = @(E) sum(gammaln(Xobsi(:)+E(:)')-gammaln(E(:)') - gammaln(alpha_0(:).*(Xobsi(:)+E(:)'))...
        + E(:)'*(alpha_0(:).*psi(alpha_0(:).*(Xobsi(:)+shape_old)) - log(2) - 0.5*(Wm(:)+Vm(:)) - log(cosh(Om(:)/2)) ));
    
    % - Free Energy
    mFE = @(E) -FE(E);
    
    % Maximize Free Energy
    shape_new= fminsearch(mFE,shape_old);
    
    % Check that optimization worked
    FE_old = FE(shape_old);
    FE_new = FE(shape_new);
    
    if (FE_old<=FE_new) && (shape_new>0)
        vi_var.shape = shape_new;
        FEE = FE_new;
    else
        warning('Shape optimization likely failed')
        FEE = FE_old;
    end
    
    % Constant part of Free Energy (ELBO)
    FE0 = sum(...
        Xobsi(:).*(Vm/2+Wm/2 - log(cosh(Om/2)))...
        - shape_old*alpha_0.*psi(alpha_0.*(Xobsi+shape_old)) ...
        + gammaln(alpha_0.*(Xobsi+shape_old)));     %-Xobsi(:)*log(2) - log(factorial(Xobsi(:)))
    
    % KLs between variational and priors
    FEKL= factors_kl(vi_param,vi_var)...
        + offset_kl(vi_param,vi_var)...
        + precision_shared_kl(vi_param,vi_var)...
        + precision_mode_kl(vi_param,vi_var);
    
    % Free Energy (ELBO)
    FEt = FEE + FE0 - FEKL;
    vi_var.FE = FEt;

    
elseif strcmp(shape_update,'fast')
    % Assume "ergodicity": match 1st Moment of the model and observation tensor
    Xhat = exp(tensor_reconstruct(vi_var.CP_mean) + vi_var.offset_mean);
    vi_var.shape = mean(Xobs(observed_id(:)))/mean(Xhat(observed_id(:)));
    
    
    
elseif strcmp(shape_update,'MM-IG')
        
    % Restrick Dataset for fast shape update
    Nlimit = 5000;
    observed_id = observed_id(randperm(length(observed_id),Nlimit));
    
    % Observed data
    Xobsi = Xobs(observed_id(:));
    
    % <W> and <W^2>
    Wm  = vi_var.tensor_mean(observed_id(:));
    Wm2 = vi_var.tensor2_mean(observed_id(:));
    
    % <V> and <V^2>
    Vm  = vi_var.offset_mean(observed_id(:));
    Vm2 = Vm.^2 + vi_var.offset_variance(observed_id(:));
    
    % <U>
    Um = vi_var.latent_mean(observed_id(:));
    
    % q(latent) ~ PG(X+shape_old,Om)
    shape_old = vi_var.shape;
    Om = sqrt(Vm2 + Wm2 + 2*Wm.*Vm);
    
    % Shape dependent terms
    FE1 = @ (E) sum(gammaln(Xobsi(:)+E(:))-gammaln(E(:)));
    FE2 = @ (E) -E(:).*sum(log(2) + 0.5*(Wm(:)+Vm(:)) + log(cosh(Om/2)));
    FE3 = @ (E)- sum(pg_kl_moment_match(...
        Xobsi(:)+shape_old,...
        Xobsi(:)+E,...
        Om(:),...
        Om(:),shape_update));
    
    FEE = @(E) FE1(E) + FE2(E) + FE3(E);
    mFE = @(E) -FEE(E);
    
    % Maximize Free Energy
    shape_new = fminsearch(mFE,shape_old);
    
    FEE_old = FEE(shape_old);
    FEE_new = FEE(shape_new);
    
    
    if (FEE_old<=FEE_new) && (shape_new>0)
        vi_var.shape = shape_new;
        FEE = FEE_new;
    else
        warning('Shape optimization likely failed')
        FEE = FEE_old;
    end
    
    
    % Likelihood part of of Free Energy (ELBO)
    FE0 = sum(Xobsi(:).*(Vm/2+Wm/2 - log(cosh(Om/2))) ); %-Xobsi(:)*log(2) - log(factorial(Xobsi(:)))
    
    % KLs between variational and priors
    FEKL= factors_kl(vi_param,vi_var)...
        + offset_kl(vi_param,vi_var)...
        + precision_shared_kl(vi_param,vi_var)...
        + precision_mode_kl(vi_param,vi_var);
    
    % Free Energy (ELBO)
    FE = FEE + FE0 - FEKL;
    vi_var.FE = FE;

elseif strcmp(shape_update,'none')
    
else
    error('The shape update method specified is not implemented')
end

end




%% KL divergence helpers 

function KL = precision_mode_kl(vi_param,vi_var)
% KL: Group Precision ARD variational and prior
if vi_param.dim_neuron>0
    
    a = vi_var.a_mode;
    b = vi_var.b_mode;
    
    a_prior = vi_var.prior_a_mode*ones(size(a));
    b_prior = vi_var.prior_b_mode*ones(size(b));
    
    KLtmp = gamma_kl(a,a_prior,b,b_prior);
    KL = sum(KLtmp(:));
    
else
    KL = 0;
end

end

function KL = precision_shared_kl(vi_param,vi_var)
% KL: Precision ARD variational and prior
if sum(vi_param.shared_precision_dim)>0
    a = vi_var.a_shared;
    b = vi_var.b_shared;
    
    a_prior = vi_var.prior_a_shared*ones(length(a),1);
    b_prior = vi_var.prior_b_shared*ones(length(a),1);
    
    KLtmp = gamma_kl(a,a_prior,b,b_prior);
    KL = sum(KLtmp(:));
    
else
    KL = 0;
end

end

function KL = offset_kl(vi_param,vi_var)
% KL: offset variational and prior
if sum(vi_param.fit_offset_dim)>0
    means = extract(vi_var.offset_mean,vi_param.fit_offset_dim);
    varis = extract(vi_var.offset_variance,vi_param.fit_offset_dim);
    preci = vi_var.offset_prior_precision;
    
    KLtmp = 0.5*varis.*preci-log(varis.*preci)+(means.^2).*preci-1;
    KL = sum(KLtmp(:));
    
else
    KL = 0;
end

end

function X =  extract(X,dim_extract)
% Extract tensor allong varying dimensions

dim_vari = find(dim_extract);
dim_fixd = find(1-dim_extract);

X = permute(X, [dim_vari,dim_fixd]);

dim_new = size(X);
dim_fin = dim_new(1:length(dim_vari));

X = X(1:prod(dim_fin));


if length(dim_fin)>1
    X = reshape(X, dim_fin);
end

end

function KL = factors_kl(vi_param,vi_var)
KL = 0;

% KL: factors variational and prior
if sum(vi_param.update_CP_dim)>0
    
    means = vi_var.CP_mean;
    varis = vi_var.CP_variance;
    preci = vi_var.CP_prior_precision;
    
    R = size(means{1},2);
    
    for dimn = 1:length(means)
        for dimi = 1:size(means{dimn},1)
            m = means{1,dimn}(dimi,:)';
            v = reshape(varis{1,dimn}(dimi,:),[R,R]);
            p = reshape(preci{1,dimn}(dimi,:),[R,R]);
            
            ds = p*v;
            
            KL = KL + 0.5*trace(ds) - log(det(ds)) + m'*p*m - R;
            
        end
    end
else
    KL = 0;
end

end

function KL = gamma_kl(alpha1,alpha2,beta1,beta2)
% KL divergence for Gamma distribution

alpha1 = alpha1(:);
alpha2 = alpha2(:);

beta1 = beta1(:);
beta2 = beta2(:);

k1 = (alpha1-alpha2).*psi(alpha1);
k2 = gammaln(alpha2)-gammaln(alpha1);
k3 = alpha2.*(log(beta1)-log(beta2));
k4 = alpha1.*(beta2-beta1)./beta1;

KL = k1 + k2 + k3 + k4;

end


function moments = pg_moment(b,c,centered)
% First moments of Polya-Gamma distribution
% For phi log laplace phi = log < exp(-ut)>
% Derive and apply at 0

if nargin <3
   centered =0; 
end

b = b(:); 
c = c(:)+eps;

% phi'(0)
phi_1 = -b.*(1./(2*c)).*tanh(c/2);

% Limits in c = 0 exist but subject to numerical precision issues.. 
l = 0.01; k = 2;
smoother = @(c,l,k) exp(-1./((c/l).^k));

% phi''(0) and phi'''(0)
P2 = @(c) (1./(4*(cosh(c/2).^2).*c.^3)).*(sinh(c)-c);
P3 = @(c) (1./(4*(cosh(c/2).^2).*c.^5)).*(c.^2.*tanh(c/2) + 3*(c-sinh(c)));

phi_2tmp = @(c) P2(c).*smoother(c,l,k) +1/24.*(1-smoother(c,l,k));
phi_3tmp = @(c) P3(c).*smoother(c,l,k) -1/60.*(1-smoother(c,l,k));

phi_2 = b.*phi_2tmp(c);
phi_3 = b.*phi_3tmp(c);


if centered
    % Outputs centered moments
    moments = [-phi_1,phi_2,-phi_3];
else
    % Outputs non centered moments
    m1 = -phi_1;
    m2 = phi_2 + phi_1.^2;
    m3 = 2*phi_1.^3 - phi_3 - 3*phi_1.*(phi_2 + phi_1.^2);
    
    moments = [m1,m2,m3];
    
end

end
