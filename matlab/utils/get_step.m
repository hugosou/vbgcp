function etaf = get_step(ite_max, rho_max,rho_min,Nperiod, tau)

if nargin<5
    tau = inf;
end


%tau = tau*ite_max; 
expd= exp(-(1:ite_max)/tau);
cosc= (1+(cos(Nperiod*pi*(1:ite_max)/ite_max)));
sinc= (sin(Nperiod*pi*(1:ite_max)/ite_max));
eta = 0.5*(cosc.*sign(sinc)+2*(sinc<0));
etaf = (rho_min+(rho_max-rho_min)*eta).*expd;

end