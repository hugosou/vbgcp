function [pdf1, pdf2, querry_left] = pg_pdf_align(pdf1,pdf2,querry1,querry2)
% PG_PDF_ALIGN efficiently aligns pdf1 and pdf2 for KL estimations
%
% INPUTS: pdf1,pdf2: Polya-Gamma densities estimated at querry1,querry2 
%         All inputs are of size upoints x bpoints
%         upoints : number of querry points (can be disjointS)
%         bpoints : number of PG densities  (can have different parameters)
%
% OUTPUTS: pdf1 and pdf2 evaluated on a joint support querry_left with same
%          sampling rate
%
% NOTE: Densities must be obtained with PG_PSD as we leverage its sampling
%       rate conventions

upoints = size(pdf1,1);
ppoints = size(pdf1,2);

assert(upoints == size(pdf2,1))
assert(ppoints == size(pdf2,2))

% Deal with linear index
offset = (0:(ppoints-1))*upoints; 

% Sampling rates
du1 = querry1(2,:)-querry1(1,:);
du2 = querry2(2,:)-querry2(1,:);

sampling_ratio = du1./du2;
id_min = (sampling_ratio>1)+1;
sampling_ratio(id_min==2) = 1./sampling_ratio(id_min==2);

% (Left/Right) = (High/Low) sampling rate. 
pdf_left = pdf1.*(id_min==1)+pdf2.*(id_min==2);
pdf_righ = pdf1.*(id_min==2)+pdf2.*(id_min==1);

clear pdf1 pdf2

% Joint support
querry_left = querry1.*(id_min==1)+querry2.*(id_min==2);
querry_righ = querry1.*(id_min==2)+querry2.*(id_min==1);

clear querry1 querry2

% Avoid boundary effects
pdf_righ = pdf_righ.*...
    (querry_righ >= min(querry_left,[],1)).*...
    (querry_righ <= max(querry_left,[],1));

% Offset for re-indexing pdf_righ
left_offset = (querry_left >= min(querry_righ,[],1));
left_offset = 1+sum(1-left_offset,1);
%left_offset = 1+find(diff(left_offset(:))==1)'-offset;

clear querry_righ

% Linear Interpolation of pdf_righ to match sampling rates
%vect_id = sampling_ratio.*(0:(upoints-1))'; % to be x2 check
%nearest_inf = 1 + floor(vect_id); 
%nearest_sup = 1 +  ceil(vect_id);
%nearest_wei = (1+vect_id-nearest_inf)./(nearest_sup-nearest_inf+eps);

vect_id = sampling_ratio.*(1:(upoints))';
nearest_inf = 1 + floor(vect_id); 
nearest_sup = min(1 +  ceil(vect_id),upoints);
nearest_wei = (1+vect_id-nearest_inf)./(nearest_sup-nearest_inf+eps);



pdf_righ = pdf_righ(nearest_inf+ offset) + (pdf_righ(nearest_sup+ offset)-pdf_righ(nearest_inf+ offset)).*nearest_wei;


clear nearest_inf nearest_sup nearest_wei


% Offset pdf_righ with proper index (out of bounds points are set to zero) 
pdf_righ(1,upoints) = 0;
full_ids = repmat((1:upoints)',[1,ppoints])-left_offset;
full_ids = (full_ids.*(full_ids>0) +upoints.*(full_ids<=0))+offset;

pdf_righ = pdf_righ(full_ids);

% Reorder matrices
pdf1 = pdf_left.*(id_min==1)+pdf_righ.*(id_min==2);
pdf2 = pdf_left.*(id_min==2)+pdf_righ.*(id_min==1);

end
