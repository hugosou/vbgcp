function [F_link, f_link, df_link] = get_f_links(model)

% Model To fit
if strcmp(model,'gaussian')
    % Gaussian
    F_link  = @(X) 0.5*(X(:)'*X(:));
    f_link  = @(X) X;
    df_link = @(X) ones(size(X));
elseif strcmp(model,'poisson')
    % Poisson
    F_link  = @(X)  sum(exp(X(:)));
    f_link  = @(X)  exp(X);
    df_link = @(X) exp(X);
elseif strcmp(model,'bernoulli')
    % Bernoulli
    F_link  = @(X)  sum(log(1+exp(X(:))));
    f_link  = @(X)  exp(X)./(1+exp(X));
    df_link = @(X)exp(2*X)./(1+exp(X)).^2;
elseif strcmp(model,'negative_binomial')    
    F_link  = [];
    f_link  = [];
    df_link = [];
else
    error('Model not supported')
end

end