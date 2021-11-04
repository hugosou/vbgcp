function [DE_0_TA,DE_0] = cp_summary(Xobs_tot,results,do_plot)


if nargin<3
    do_plot=0;
end

Loss_tot = results.fit.Loss_tot;
CP = results.fit.GCP;
R   = results.fit_param.R;


if do_plot
    
    figure
    scatter(1:length(Loss_tot), Loss_tot,30, 'k', 'filled')
    ylabel(['a.u'])
    xlabel(['Iterattion'])
    title('Loss')
    box on; axis tight
    
    
    figure
    for dimi=1:size(CP,2)
        for order=1:R
            subplot(size(CP,2),R, order+(dimi-1)*R)
            scatter(1:size(CP{1,dimi},1),CP{1,dimi}(:,order), 30,'k','filled')
            box on;
            if order==1
                ylabel(['Dimension ', num2str(dimi)])
            end
            
            if dimi==1
                title(['Component ', num2str(order), '/R'])
            end
        end
    end
    
end

if nargout>0
    
    Xhati =  results.fit.What+results.fit.offset;
    Xhat  =  results.fit_param.f_link(Xhati);
    
    Xhat_trialavg = mean(Xhat,length(size(Xobs_tot)));
    Xobs_trialavg = mean(Xobs_tot,length(size(Xobs_tot)));
    
    if length(size(Xhat))==(length(size(Xobs_tot))-1)
        
        Xhat2 = repmat(Xhat,[ones(1,length(size(Xhat))),size(Xobs_tot,length(size(Xobs_tot)))]);
        
        
        
        
        D    = deviance_poisson(Xobs_tot, Xhat2);
        X0     = repmat(mean(Xobs_tot(:)),size(Xobs_tot));
        DX0    = deviance_poisson(Xobs_tot, X0);
    else
        D    = deviance_poisson(Xobs_tot, Xhat);
        X0     = repmat(mean(Xobs_tot(:)),size(Xhat));
        DX0    = deviance_poisson(Xobs_tot, X0);
    end
    
    D_TA   = deviance_poisson(Xobs_trialavg, Xhat_trialavg);
    DX0_TA = deviance_poisson(Xobs_trialavg, mean(X0,4));
    
    DE_0    = 1 - D/DX0;
    DE_0_TA = 1 - D_TA/DX0_TA;
    
end

end