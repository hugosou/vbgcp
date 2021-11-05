function vi_var = vi_update_linear_response(Xobs,vi_var,vi_param)
% Use Linear Reponse Method for more accurate covariance estimate of GCP
% decomposition
% Adapted from Giordano et al. (2015).

% Dimensions of the problem
Xdims = size(Xobs);
R = size(vi_var.CP_mean{1},2);

% Linear Response Matrix H = d2L / dmdm'
H_linear_response = get_linear_response(Xobs,vi_var,vi_param);

% Full Variational Covariance (Block Diagonal)
Vhat = full_ss_covariance(vi_var.CP_mean, vi_var.CP_variance);
Vhat = nearestSPD(Vhat);

% Update Covariance
Vnew = (eye( R*(R+1)*sum(Xdims)) - Vhat*H_linear_response) \ Vhat;

Vnew = 0.5*(Vnew+Vnew');
if min(eig(Vnew))<-(1e-6)
   warning('Corrected Covariance not semi-positive definite. VI likeliy did not converge') 
end


% Marginalize Full Covariance
CP_variance_new = marginalize_CP_variance(Vnew,R,Xdims);

% Store
vi_var.CP_variance = CP_variance_new;

figure; hold on
subplot(2,1,1);hold on
plot(diag(Vhat))
plot(diag(Vnew))

subplot(2,1,2);hold on
plot(diag(Vnew)-diag(Vhat))




end

function CP_variance = marginalize_CP_variance(Vtot,R,Xdims)
% Marginalize the full covariance of G-CP decomposition variational mean and Variance

% Init
CP_variance = cellfun(@(Z) zeros(Z,R*R), num2cell(Xdims), 'UniformOutput', false);

% Extract and store Block Diagonal Covariances
block_size = R*(R+1);
for dimn = 1:length(Xdims)
    for dimi = 1:Xdims(dimn)
        % Block Ids
        id_offset = sum(Xdims(1:(dimn-1)));
        id_bloc   = dimi + id_offset;
        id_bloc_cur = (id_bloc-1)*block_size+(1:block_size);
        
        Vtmp = Vtot(id_bloc_cur,id_bloc_cur);
        Vtmp = reshape(Vtmp(1:R,1:R),1,R*R);
        CP_variance{1,dimn}(dimi,:) = Vtmp;
    end
end

end

function H_linear_response = get_linear_response(Xobs,vi_var,vi_param)
% Outputs linear response matrix H = d2L / dmdm'
% Where: - L = <log p(vi_var | Xobs)>_q(vi_var) 
%        - vi_var contains the moments of Generalized-CP decomposition 
%        - q is the variational distribution
% Implemented from Giordano et al. (2015).

% Variational Moments E(x, Vect(xx'))
m_tot = vi_var.CP_mean;
S_tot = get_AAt(vi_var.CP_mean,vi_var.CP_variance);

% Other Variational Parameters
Ulatent = vi_var.latent_mean;
Voffset = vi_var.offset_mean;
Eshape  = vi_var.shape;

% Dimensions of the problem
R = size(m_tot{1},2);
Xdims = size(Xobs);
block_size = R*(R+1);

% Deals with missing entries
observed_data = vi_param.observed_data;
if all(observed_data(:))
    % All data are observed
    observed_data =ones(Xdims);
end

H_linear_response = zeros( R*(R+1)*sum(Xdims) );

for dimn_1 = 1:length(Xdims)
    
    % Grasp Useful Tensors
    Xn = tensor_unfold(Xobs,dimn_1);
    Un = tensor_unfold(Ulatent,dimn_1);
    Vn = tensor_unfold(Voffset,dimn_1);
    On = tensor_unfold(observed_data,dimn_1);
    Zn = (Xn - Eshape)./(2*Un+eps) - Vn;
    
    for dimi_1 = 1:Xdims(dimn_1)
        % Horizontal Coordinate
        id_offset_1 = sum(Xdims(1:(dimn_1-1)));
        id_bloc_1   = dimi_1 + id_offset_1;
        ids1_block_cur = (id_bloc_1-1)*block_size+(1:block_size);
        
        % Keep indices corresponding to observed data
        Oni     = find(On(dimi_1,:));
        
        % <Z> and <U> with only the relevant index
        Zni = Zn(dimi_1,Oni);
        Uni = Un(dimi_1,Oni);
        Yni  = Zni.*Uni;
        
        for dimn_2= 1:length(Xdims)
            for dimi_2 = 1:Xdims(dimn_2)
                % Vertical Coordinate
                id_offset_2 = sum(Xdims(1:(dimn_2-1)));
                id_bloc_2   = dimi_2 + id_offset_2;
                ids2_block_cur = (id_bloc_2-1)*block_size+(1:block_size);
                
                % H is symetric with block diagonal = 0
                if ids2_block_cur > ids1_block_cur
                    
                    % First moments : d2L/dm1dm2
                    dB = delta_b(m_tot,dimn_1,1,dimi_2);
                    dB = dB(Oni,:);
                    d2L_m1m2 = diag(Yni*dB);
                    
                    % Second moments : d2L/dS1dS2
                    dBB = delta_b(S_tot,dimn_1,1,dimi_2);
                    dBB = dBB(Oni,:);
                    d2L_S1S2 = -0.5*diag(Uni*dBB);
                    
                    % Curren Block
                    H_12 = [[d2L_m1m2, zeros(R,R*R)];
                        [zeros(R*R,R), d2L_S1S2]];
                    
                    H_linear_response(ids1_block_cur,ids2_block_cur) = H_12;
                    H_linear_response(ids2_block_cur,ids1_block_cur) = H_12';
                    
                end
                
            end
            
        end
        
    end
    
end

end

function dB = delta_b(means,dimn_1,dimn_2,dimi_2)
% Helper to differentiate Khatri-Raoh product
R = size(means{1},2);

m0 = means;
m1 = means;
m0{1,dimn_2}(dimi_2,:) = zeros(1,R);
m1{1,dimn_2}(dimi_2,:) =  ones(1,R);
B0 = KhatriRaoProd_mean(m0,dimn_1);
B1 = KhatriRaoProd_mean(m1,dimn_1);
dB = B1 - B0;

end

function Vtot = full_ss_covariance(CP_mean, CP_variance)
% Full covariance of G-CP decomposition variational mean and Variance

% Problem dimensions
R = size(CP_mean{1},2);
Xdims = cellfun(@(Z) size(Z,1), CP_mean);

% Build full covariance matrix
block_size = R*(R+1);
Vtot = zeros(block_size*sum(Xdims));

for dimn = 1:length(Xdims)
    for dimi = 1:Xdims(dimn)
        % Block Ids
        id_offset = sum(Xdims(1:(dimn-1)));
        id_bloc   = dimi + id_offset;
        id_bloc_cur = (id_bloc-1)*block_size+(1:block_size);
        
        % Build Sufficient Statistics Covariances
        m_in = CP_mean{1,dimn}(dimi,:);
        S_in = CP_variance{1,dimn}(dimi,:);
        V_in = ss_covariance(m_in(:),reshape(S_in,R,R));
        
        
        [~, cc] = chol(V_in);
        
        cccheck(id_bloc,id_bloc) = 1;
        
        
        Vtot(id_bloc_cur,id_bloc_cur) = V_in;
    end
end

end

function V = ss_covariance(m,S)
% Covariance matrix of MVN Sufficient Statistic

R = size(m,1);
assert(all(R==size(S)))


Cxx = S;

CxX = kron(S,m')+kron(m',S);
% CxX = zeros(R,R*R);
% for i =1:R
%     for j1=1:R
%         for j2=1:R
%             jj = (j1-1)*R+j2;
%             CxX(i,jj) = S(i,j1)*m(j2) + S(i,j2)*m(j1);
%         end
%     end
% end

CXX = zeros(R*R,R*R);
for i1 =1:R
    for i2 =1:R
        ii = (i1-1)*R+i2;
        % vertical axis
        
        for j1=1:R
            for j2=1:R
                jj = (j1-1)*R+j2;
                % horizontal axiz
                
                CXX(ii,jj) =m(i1)*m(j1)*S(i2,j2)...
                    +m(i1)*m(j2)*S(i2,j1)...
                    +m(i2)*m(j1)*S(i1,j2)...
                    +m(i2)*m(j2)*S(i1,j1)...
                    + S(i1,j1)*S(i2,j2)  ...
                    + S(i1,j2)*S(i2,j1);            
            end
        end
    end
end


V = [[Cxx,CxX]; [CxX', CXX]];

% In case of rounding errors
V = nearestSPD(V);

end

function Ahat = nearestSPD(A)
% nearestSPD - the nearest (in Frobenius norm) Symmetric Positive Definite matrix to A
% usage: Ahat = nearestSPD(A)
%
% From Higham: "The nearest symmetric positive semidefinite matrix in the
% Frobenius norm to an arbitrary real matrix A is shown to be (B + H)/2,
% where H is the symmetric polar factor of B=(A + A')/2."
%
% http://www.sciencedirect.com/science/article/pii/0024379588902236
%
% arguments: (input)
%  A - square matrix, which will be converted to the nearest Symmetric
%    Positive Definite Matrix.
%
% Arguments: (output)
%  Ahat - The matrix chosen as the nearest SPD matrix to A.
if nargin ~= 1
  error('Exactly one argument must be provided.')
end
% test for a square matrix A
[r,c] = size(A);
if r ~= c
  error('A must be a square matrix.')
elseif (r == 1) && (A <= 0)
  % A was scalar and non-positive, so just return eps
  Ahat = eps;
  return
end
% symmetrize A into B
B = (A + A')/2;
% Compute the symmetric polar factor of B. Call it H.
% Clearly H is itself SPD.
[U,Sigma,V] = svd(B);
H = V*Sigma*V';
% get Ahat in the above formula
Ahat = (B+H)/2;
% ensure symmetry
Ahat = (Ahat + Ahat')/2;
% test that Ahat is in fact PD. if it is not so, then tweak it just a bit.
p = 1;
k = 0;
while p ~= 0
  [R,p] = chol(Ahat);
  k = k + 1;
  if p ~= 0
    % Ahat failed the chol test. It must have been just a hair off,
    % due to floating point trash, so it is simplest now just to
    % tweak by adding a tiny multiple of an identity matrix.
    mineig = min(eig(Ahat));
    Ahat = Ahat + (-mineig*k.^2 + eps(mineig))*eye(size(A));
  end
end
end



