function Asigned = signorder_cp(A,dimi)


Asigned = cell(1,size(A,2));
R = size(A{1,1},2); 


for rr = 1:R
    
    si_RR = 1;
    
    for ii=1:size(A,2)
        Ai_rr = A{1,ii}(:,rr);
        si_rr = sign(sum(Ai_rr));
        Ai_rr = Ai_rr.*si_rr;
        si_RR = si_RR.*si_rr;
        
        Asigned{1,ii}(:,rr) = Ai_rr;
    end
    
    Asigned{1,dimi}(:,rr) = Asigned{1,dimi}(:,rr)*si_RR;
    
end



end