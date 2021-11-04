function vi_var = vi_update_latent(vi_var,vi_param,Xobs)
% Variational update of the PG latent 

% Grasp Current Sample
Eshape  = vi_var.shape;

% First moments
tensor =  vi_var.tensor_mean;
offset =  vi_var.offset_mean;

% Second moments
tensor2 = vi_var.tensor2_mean;
offset2 = vi_var.offset_variance + vi_var.offset_mean.^2;

observed_data = vi_param.observed_data;
if all(observed_data(:)==1)
    % All data are observed
    observed_id = (1:numel(Xobs))';
else
    % There is missing data. 
    % We only sample observed ones.
    observed_id = find(observed_data);
end

% sqrt(<(W +V)^2>) 
omega = sqrt(offset2(observed_id) + tensor2(observed_id) + 2.*tensor(observed_id).*offset(observed_id));
Xtmp  = Xobs(observed_id);

% Update only relevant means
latent = zeros(numel(Xobs),1);
latent_mean_tmp = ((Eshape + Xtmp)./(2*omega)) .*tanh(omega/2);
latent(observed_id) = latent_mean_tmp;

% Reshape Latents
latent = reshape(latent, size(Xobs));

% Update variables
vi_var.latent_mean = latent;


end