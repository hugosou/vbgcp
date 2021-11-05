function BBn = KhatriRaoProd_var(CP_mean,CP_variance,dimn)
% INPUTS : CP decomposition mean    : {E(A1), ..., E(AD)}
%          CP decomposition variance: {V(A1), ..., V(AD)}
%          E(An)(i,:) = m_in          ~ 1 x R
%          V(An)(i,:) = Vec(Sigma_in) ~ 1 x RR
%
% OUPUTS : BBn = AA_D * ... * AA_n+1 * AA_n-1 * ... * AA_1
%          AAn(i,:) = Vec(m_in'm_in + Sigma_in)  ~ 1 x RR

% Build AAT
factors = get_AAt(CP_mean,CP_variance);

% Helper matrix of size prod_j(I_i) x (RR)
BBn = KhatriRaoProd_mean(factors,dimn);

end