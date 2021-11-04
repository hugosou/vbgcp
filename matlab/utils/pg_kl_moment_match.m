function KL = pg_kl_moment_match(b1,b2,c1,c2,str_option)
% Approximate KL Divergence between Polya-Gamma Distribution
% (i)  Moment match PG(bi,ci) distributions to (inverse) Gamma.
% (ii) Theoretical KL for gamma distributions.

if nargin <5
   str_option = 'MM-G'; 
end

if strcmp(str_option,'MM-IG')
    % Fit an inverse gamma distribution
    
    % Moments 1
    [alpha1,beta1] = pg_to_ig(b1(:),c1(:));
    
    % Moments 2
    [alpha2,beta2] = pg_to_ig(b2(:),c2(:));
    
elseif strcmp(str_option,'MM-G')
    % Fit a gamma distribution
    
    % Moments 1
    [alpha1,beta1] = pg_to_g(b1(:),c1(:));
    
    % Moments 2
    [alpha2,beta2] = pg_to_g(b2(:),c2(:));
   
end
    


% KL Divergence
KL = mm_kl(alpha1,alpha2,beta1,beta2);

end

function KL = mm_kl(alpha1,alpha2,beta1,beta2)

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

function [alpha,beta] = pg_to_ig(b,c)
% Fit Inverse Gamma to PG(b,c) using moment matching

% Moment of the PG distribution
mtot = pg_moment(b,c);

m1 = mtot(:,1);
m2 = mtot(:,2);

alpha = (2*m2-m1.*m1)./(m2-m1.*m1);
beta  = (m1.*m2)./(m2-m1.*m1);

end

function [alpha,beta] = pg_to_g(b,c)
% Fit Gamma to PG(b,c) using moment matching

% Moment of the PG distribution
mtot = pg_moment(b,c);

m1 = mtot(:,1);
m2 = mtot(:,2);

m = m1;
v = m2-m1.^2;

alpha = m.^2./v;
beta  = m./v;

end


