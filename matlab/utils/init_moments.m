function moments = init_moments(Adims,R)

% Init Full  CP decomposition
mt  = cell(1,length(Adims));
vt  = cell(1,length(Adims));
gt  = cell(1,length(Adims));

% Initialization
for dimi=1:length(Adims)
    mt{1,dimi} = zeros(Adims(1,dimi),R);
    vt{1,dimi} = zeros(Adims(1,dimi),R);
    gt{1,dimi} = zeros(Adims(1,dimi),R);
end

moments = struct();
moments.mt=mt;
moments.vt=vt;
moments.gt=gt;

end