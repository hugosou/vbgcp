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