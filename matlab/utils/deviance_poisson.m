function D = deviance_poisson(Xobs, Xhat)
    % D = 2*sum(sum(sum(  Xobs.*log(eps + Xobs./(Xhat+eps)) - Xobs + Xhat)));
    D = 2 * sum(Xobs(:).*log(eps+Xobs(:)./(Xhat(:)+eps)) - Xobs(:) + Xhat(:));
end


