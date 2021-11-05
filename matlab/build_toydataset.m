function [Xobs,observed_data,true_params,Xtrue] = build_toydataset(model_true,Rtrue,Xdims,add_offset,add_missing)
% Generate a 5D count tensor of rank R = 4

% Simulated parameters
true_params = struct();

% Dimensions of the problem
D1 = Xdims(1);
D2 = Xdims(2);
D3 = Xdims(3);
D4 = Xdims(4);
D5 = Xdims(5);

% Time Dynamics
Ttrue = zeros(D2,Rtrue);
Ttrue(:,1) = sin(linspace(0,2*pi,D2));
Ttrue(:,2) = cos(linspace(0,2*pi,D2));
Ttrue(:,3) = abs(sin(linspace(0,2*pi,D2))).^2;
Ttrue(:,4) = abs(sin(linspace(0,2*pi,D2))).^2;

% Neuron Loadings
Ntrue = normrnd(0,2,D1,Rtrue);

% Neuron Groups
neurons_groups = zeros(D1,3);
neurons_groups(:,1) = (1:D1) < D1/3;
neurons_groups(:,2) = (D1/3 <= (1:D1)).* ((1:D1)< 2*D1/3);
neurons_groups(:,3) = 2*D1/3 <= (1:D1);
CG =[sum(neurons_groups.*[0,1,1],2),sum(neurons_groups.*[1,0,1],2),sum(neurons_groups.*[0,1,0],2),sum(neurons_groups.*[1,1,1],2)];
Ntrue=Ntrue.*CG;
Ntrue(:,3) = Ntrue(:,4); 

% Condition Dependent
Ctrue = zeros(D3,Rtrue);
Ctrue(:,1) = [1;1;1];
Ctrue(:,2) = [0.2;0;1];
Ctrue(:,3) = [1;0;1];
Ctrue(:,4) = [0;1;0];

% Experiment Dependent
Etrue = ones(D4,Rtrue)+randn(D4,Rtrue);

% Trial Dependent
Ktrue = zeros(D5,Rtrue);
Ktrue(:,1) = linspace(0,1,D5);
Ktrue(:,2) = linspace(1,0,D5);
Ktrue(:,3) = linspace(0,1,D5).^2;
Ktrue(:,4) = ones(1,D5);

% Low dim Tensor
CPtrue = cell(1,5);
CPtrue{1,1} = Ntrue;
CPtrue{1,2} = Ttrue-mean(Ttrue,1);
CPtrue{1,3} = Ctrue;
CPtrue{1,4} = Etrue;
CPtrue{1,5} = 0.1*Ktrue;

if strcmp(model_true,'poisson')
    CPtrue{1,5} = 0.5*Ktrue;
end

% Reconstruct Dynamics
Wtrue = tensor_reconstruct(CPtrue);

% Add the offset allong fit_offset_dim dimensions
fit_offset_dim = [1,0,1,0,0];
vtrue = rand((fit_offset_dim).*size(Wtrue)+not(fit_offset_dim));
vtrue = 0.1*vtrue;

offset = add_offset*repmat(vtrue, fit_offset_dim + not(fit_offset_dim).*size(Wtrue));
Wtrue = Wtrue + offset;

% Noise model
[~, f_link_true, ~] = get_f_links(model_true);
true_params.f_link_true = f_link_true;

if strcmp(model_true,'gaussian')
    sig_noise =0.1;
    true_params.sig_noise=sig_noise;
    Xobs = f_link_true(Wtrue) + normrnd(0,sig_noise,size(Wtrue));
    Xtrue= f_link_true(Wtrue);
    
elseif strcmp(model_true,'poisson')
    Xobs  = poissrnd(f_link_true(Wtrue));
    Xtrue= f_link_true(Wtrue);
    
elseif strcmp(model_true,'negative_binomial')
    %Pd = 1./(1+exp(-Wtrue)); 
    shape = 80;
    true_params.shape  = shape;
    Pd = 1./(1+exp(Wtrue)); % Definition differs from paper because of matlab conv.
    %Pd = 1./(1+exp(-Wtrue));
    
    Rd = shape*ones(size(Pd));
    Xobs = nbinrnd(Rd,Pd);
    Xtrue= exp(Wtrue)*shape;
    
elseif strcmp(model_true,'bernoulli')
    Xobs  = rand(size(Wtrue))<f_link_true(Wtrue);
    Xtrue = Wtrue;
else
    error('Model not supported')
end

% Model Missing Entries
if add_missing
    neuron_expt = [1,floor(D1/6),floor(D1/2), floor(3*D1/4),  D1];
    observed_data = zeros(D1,D4);
    observed_data((neuron_expt(1)+0):neuron_expt(2),1) = 1;
    observed_data((neuron_expt(2)+1):neuron_expt(3),2) = 1;
    observed_data((neuron_expt(3)+1):neuron_expt(4),3) = 1;
    observed_data((neuron_expt(4)+1):neuron_expt(5),4) = 1;
    observed_data = repmat(observed_data, [1,1, size(Xobs,2),size(Xobs,3),size(Xobs,5)]);
    observed_data = permute(observed_data , [1,3,4,2,5]);
else
    observed_data = 1;
end

% Final Observed Tensor
Xobs = Xobs.*observed_data;

true_params.CPtrue = CPtrue;
true_params.offset = offset;
true_params.fit_offset_dim = fit_offset_dim;
true_params.neurons_groups = neurons_groups;








