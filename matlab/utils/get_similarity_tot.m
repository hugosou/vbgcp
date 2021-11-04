function [smlty_tot_noref,ref_tot,permt_tot,smlty_tot,sig_tot] = get_similarity_tot(models,used_dims,refid, combined_dims,observed_data)
% Get similarity index from a grid search with d1 x d2 x ... x dL params
% with Ntest tests.
% models_tot is a d1 x d2 x ... x dL x Ntests cell containing CP
% decompositions

mdims = ndims(models);
msize = size(models);

Ntest = msize(end);
Ngrid = prod(msize(1:end-1));

% Use linear index to use flexible sizes for models
linear_index_tot = reshape(1:prod(msize),msize);

linear_index_tot = permute(linear_index_tot, [mdims,1:mdims-1]);
linear_index_tot = reshape(linear_index_tot,[Ntest,Ngrid]);

if nargin<3
    refid = 1:Ntest;
end

if nargin<2
   used_dims = ones(1,size(models{1},2));
end

if nargin<4
   combined_dims = 0;
   observed_data = 1;
end

smlty_tot = zeros(Ngrid,Ntest);
ref_tot   = zeros(Ngrid,1);
permt_tot = cell(msize(1:end-1));
sig_tot   = cell(msize(1:end-1));



for ngrid=1:Ngrid
    disp(['similarities: ',num2str(ngrid),'/',num2str(Ngrid)])
    
    % Gather models with common parameters
    models_cur = cell(1,Ntest);
    for ntest = 1:Ntest
        linear_index_cur = linear_index_tot(ntest,ngrid);
        models_cur{1,ntest} = models{linear_index_cur};
    end
    
    
    if sum(cellfun(@(Z) isempty(Z), models_cur))==0
        
        % Estimate similarities using a range of reference
        best_smlty = zeros(1,Ntest);
        best_permt = 1;
        best_ref = 1;
        best_sig = 1;
        for ref=refid
            idred=1:Ntest; idred(ref) = [];
            if not(sum(combined_dims)==2)
                [smlty,permt,signt] = get_similarity(models_cur, ref,used_dims);
            else
                [smlty,permt,signt] = get_similarity_missing(models_cur, ref,observed_data,combined_dims);
            end

            if median(best_smlty(idred))<median(smlty(idred))
                best_smlty = smlty;
                best_permt = permt;
                best_ref   = ref;
                best_sig   = signt;
            end
        end
        
        % Gather Results
        smlty_tot(ngrid,:) = best_smlty;
        permt_tot{ngrid}   = best_permt;
        ref_tot(ngrid,1)   = best_ref;
        sig_tot{ngrid}     = best_sig;
        
    else
        warning('Some Empty models')
        smlty_tot(ngrid,:) = zeros(1,Ntest);
        permt_tot{ngrid}   = [];
        ref_tot(ngrid,1)   = 1;
        sig_tot{ngrid}     = [];
    end
    
end

% Reorder Results



% Remove the reference in the similarities
smlty_tot_noref = zeros(size(smlty_tot,1),size(smlty_tot,2)-1);
for ngrid=1:Ngrid
    ref = ref_tot(ngrid);
    idred=1:Ntest; idred(ref) = [];
    smlty_tot_noref(ngrid,:) = smlty_tot(ngrid,idred);
end



smlty_tot_noref = reshape(smlty_tot_noref', [Ntest-1, msize(1:end-1)]);
smlty_tot_noref = permute(smlty_tot_noref, [2:ndims(smlty_tot_noref),1]);

smlty_tot = reshape(smlty_tot', [Ntest, msize(1:end-1)]);
smlty_tot = permute(smlty_tot, [2:ndims(smlty_tot),1]);
if length(msize(1:end-1))>1
    ref_tot   = reshape(ref_tot, msize(1:end-1));
end





%     function y = idten(nsizes,x)
%         y = cell(1,length(nsizes));
%         [y{:}] = ind2sub(nsizes,x); y = cell2mat(y);
%     end
%
%     function x = idlin(nsizes, y)
%         x = 1 + [1,cumprod(nsizes(1:(end-1)))]*(y-1)';
%     end


end



