function KL = pg_kl_moment_match_fast(b1,b2,c1,c2,option)
% Approximate KL Divergence between Polya-Gamma Distribution
% (i)  Moment match PG(bi,ci) distributions to (inverse) Gamma.
% (ii) Theoretical KL for gamma distributions.

if nargin <5
   option = 'MM-G'; 
end

if strcmp(option,'MM-IG')
    % Fit an inverse gamma distribution
    
    % Moments 1
    [alpha1,beta1] = pg_to_ig(b1(:),c1(:));
    
    % Moments 2
    [alpha2,beta2] = pg_to_ig(b2(:),c2(:));
    
elseif strcmp(option,'MM-G')
    % Fit a gamma distribution
    [alpha0,beta0] = pg_to_g(b1(:),c1(:));
    
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


function moments = pg_moment(b,c)
% First moments of Polya-Gamma distribution
% For phi log laplace phi = log < exp(-ut)>
% Derive and apply at 0

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

% Associated Moments
m1 = -phi_1;
m2 = phi_2 + phi_1.^2;
m3 = 2*phi_1.^3 - phi_3 - 3*phi_1.*(phi_2 + phi_1.^2);

moments = [m1,m2,m3];

end

