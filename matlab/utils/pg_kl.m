function KL  = pg_kl(pdf1,pdf2,querry1,querry2)
% PG_KL align and estimate KL divergence from pdf1 and pdf2
%
% INPUTS: pdf1,pdf2: Polya-Gamma densities estimated at querry1,querry2
%         All inputs are of size upoints x bpoints
%         upoints : number of querry points (can be disjointS)
%         bpoints : number of PG densities  (can have different parameters)
%
% NOTE: Densities must be obtained with PG_PSD as we leverage its sampling
%       rate conventions


% Align and match sampling rate of pdf1 and pdf2 
[pdf1_bis, pdf2_bis, querry] = pg_pdf_align(pdf1,pdf2,querry1,querry2);

pdf1 = abs(pdf1);
pdf1_bis = abs(pdf1_bis);
pdf2_bis = abs(pdf2_bis);

% Shared sampling rate 
du = querry(2,:)-querry(1,:);

% Sampling rate from base pdf
du1 = querry1(2,:)-querry1(1,:);

% Entropy of the first densities
qlogq = sum(pdf1.*log(pdf1+eps),1).*du1;

% Cross entropy on joint support
qlogp_joint    = sum(pdf1_bis.*log(pdf2_bis+eps),1).*du;

% Cross entropy on disjoint support (thy infinite, simple estimation)
disjoint_1 = 1-(querry1 >= min(querry2,[],1)).*(querry1 <= max(querry2,[],1));
qlogp_disjoint = sum(disjoint_1.*pdf1.*log(0*disjoint_1.*pdf1+eps),1).*du1;
%qlogp_disjoint = qlogp_disjoint.*10e30;

% Full KL
KL = qlogq - qlogp_joint - qlogp_disjoint;

end