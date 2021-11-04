function [pdf,uvec] = pg_psd(b,c,upoints,nstd,log_base)
% Probability density function of Polya-Gamma PG(b,c) estimated at uvec
%
%   - PG_PSD: uses a truncated sum to approximate pdf arround its mean and
%             variance (accessible through PG_MOMENTS). 
%   
%   - CAUTION: numerical errors quickly make this sum diverge for b > 20 
%              we hence leverage the convolution property of PG:
%              if u1 ~ PG(b1,c1), u2 ~ PG(b2,c2), u1+u2 ~ PG(b1+b2,c1+c2) 
%
%   - INPUTS: b and c, parameters of PG distribution
%             upoints, number of points at which pdf are estimated
%             nstd, interval size (in std unit) around mean to evaluate pdf
%             log_base, decomposition use for self convolution
% H.Soulat (2021).

b = b(:)';
c = c(:)';

assert(length(b) == length(c));



if nargin <3
    % Number of points at which pdf are estimated
    upoints = 1000;
end

if nargin <4
    % Num Std Interval around PG mean
    nstd = 10;
end

if nargin <5
    % Base used to perform self convolution
    log_base = 10;
end

% Number of convolution necessary for each b
b_log = floor(log(b)/log(log_base));
b_cur = b_log.*(b_log>0);

% Initial estimate
b_min = b./(log_base.^b_log);
[pdf, uvec] =  pg_thr_psd(b_min,c,upoints,nstd);

% Convolution loop
id_cur = find(b_cur);

while not(isempty(id_cur))
    
    % To be convolved
    pdf_cur = pdf(:,id_cur);
    querry_cur = uvec(:,id_cur);

    % Convolution
    [pdf_cur, querry_cur] = self_conv(pdf_cur,log_base*ones(size(pdf_cur,2),1),querry_cur,'same', nstd);
    
    % Store
    pdf(:,id_cur) = pdf_cur;
    uvec(:,id_cur) = querry_cur;

    b_cur(id_cur) =  b_cur(id_cur)-1;
    id_cur = find(b_cur);
end

end


function [pdf_new,xvec_new]=self_conv(pdf,nconv,xvec,shape,nstd)
% Self convolve pdf nconv time using FFT
% Initial densities are evaluated at xvec
% shape and nstd determine the size of new evaluations

if nargin <4
   % Prune observations
   shape = 'same'; 
end

if nargin <5
    % New interval around std
   nstd = 5; 
end

% Original Step size
du = xvec(2,:)-xvec(1,:);

% Querry size
Lx = size(pdf,1);

% Numbers of convolution
Ln = size(nconv,1);
assert(Ln==size(pdf,2));

% New vector size
Ly = max(nconv(:))*size(pdf,1)-1;

% Find smallest power of 2 that is > Ly
Ly2= pow2(nextpow2(Ly));    

% Fast Fourier transform
pdf=fft(pdf, Ly2,1);	

% F(f*f*...*f) = F(f)*F(f)*...*F(f)
Y=pdf.^repmat(nconv',[Ly2,1]);        	           % 

% Inverse fast Fourier transform
pdf_new = real(ifft(Y, Ly2,1));      

% Resize pdf
pdf_new = pdf_new.*du.^(nconv(:)'-1);

% New querry points
xvec_new = xvec(1,:) + (0:(Ly2-1))'*du;


% Keep only samples around peak of the distribution
if strcmp(shape,'same')
    
    % Neep every 'step_point'
    step_point     = floor(sqrt(nstd *nconv(:)'));
    
    % Max of the distribution
    [~,maxi_point] = max(pdf_new,[],1);
    
    % Ids to be kept
    new_min_u = max(1,(maxi_point - floor((step_point).*(Lx)/2)));
    new_id = new_min_u+(0:(Lx-1))'.*step_point;
    new_id = new_id + ((1:Ln)-1)*Ly2;
    new_id = new_id(:);
    
    % Prune and reshape outputs
    pdf_new = pdf_new(new_id);
    pdf_new = reshape(pdf_new,[Lx,Ln]);
    
    xvec_new = xvec_new(new_id);
    xvec_new = reshape(xvec_new,[Lx,Ln]);
    
end

end

function [pdf,xvec] = pg_thr_psd(b,c,upoints,nstd)
% Polya-gamma PG(b,c) probability density function.
% Evaluated at xvec
% CAUTION: For b > 30 numerical errors diverges
%          Need to use convolutions

% Numbert of evaluation points
if nargin <3
    upoints = 1000;
end

% Numbert of evaluation points
if nargin <4
    nstd = 5;
end

% Moment of PG(b(:),c(:)) used for evaluation intervals
moments_tmp = pg_moment(b,c);
m = moments_tmp(:,1);
v = moments_tmp(:,2) - moments_tmp(:,1).^2;

% Evaluation intervals
umin = max(eps, m- nstd*sqrt(v));
umax = m + nstd*sqrt(v);
xvec = (umin + (0:(upoints-1)).*(umax-umin)./upoints)';

% Truncate infinite sum depending on b
nmax = 100*floor(max(b(:)))-1; % 200 ?
nvec = reshape(0:nmax,[1,1,nmax+1]);
bvec = b(:)';
cvec = c(:)';

% Infinite sum terms (n,b)
log_nb = (bvec-1)*log(2)...
    +gammaln(nvec+bvec)...
    -gammaln(nvec+1)...
    -gammaln(bvec) ...
    +log(2*nvec + bvec);

% Infinite sum terms (n,b,u)
log_nbu = log_nb - ((2*nvec+bvec).^2)./(8.*xvec) - (3/2)*log(xvec) ;
log_nbu = reshape(permute(log_nbu,[3,1,2]),nmax+1, upoints*length(b(:)));

% Alternating sum
m1n = (-1).^nvec(:);
pdf = m1n'*exp(log_nbu);

% PG(b,0): Reshape And Normalize
pdf = reshape(pdf, [upoints, length(b(:))]);
pdf = pdf./sqrt(2*pi);

% PG(b,c): Add Parameter c
pdf = (cosh(0.5*cvec).^bvec).*exp(-0.5*xvec.*cvec.^2).*pdf;

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
