function plot_similarities(similarilties,deviances,model_sizes,model_penalties)

%% Plot results


figure
pp = [];qq = [];

refSim = squeeze(mean(mean(similarilties,2),3));
refDev = squeeze(mean(mean(deviances,2),3));


RLtest = size(similarilties,1);
NLtest = size(similarilties,2);

for Ltestid = 1:NLtest
    
    smltycur = squeeze(similarilties(:,Ltestid,:));
    deviacur = squeeze(deviances(:,Ltestid,:));
    
    mu = mean(smltycur,2);
    mup = mu+std(smltycur,[],2);
    mum = mu-std(smltycur,[],2);
    xx = model_sizes;
    
    nu = median(deviacur,2);
    nup = nu+std(deviacur,[],2);
    num = nu-std(deviacur,[],2);
    
    p=subplot(NLtest,2,1+2*(Ltestid-1));hold on
    plot(xx,refSim, 'linewidth',1.5,'color','g','linestyle','-.')
    l1=plot(xx,mu, 'linewidth',1.5,'color','r');
    scatter(xx,mu,20,'k','filled')
    patch([xx(:); flipud(xx(:))], [mup(:); flipud(mum(:))], 'k', 'FaceAlpha',0.2)
    pp=[pp,p];
    box on; xlim([xx(1),xx(end)])
    ylim([0 1])
    
    legend(l1,{['\lambda=', num2str(model_penalties(1,Ltestid),2)]})
    
    if Ltestid==1
        title('Similarity')
    end
    
    if Ltestid==NLtest
        xlabel('R')
    end
    
    q=subplot(NLtest,2,2+2*(Ltestid-1));hold on
    plot(xx,refDev, 'linewidth',1.5,'color','g','linestyle','-.')
    plot(xx,nu, 'linewidth',1.5,'color','r')
    scatter(xx,nu,20,'k','filled')
    patch([xx(:); flipud(xx(:))], [nup(:); flipud(num(:))], 'k', 'FaceAlpha',0.2)
    qq=[qq,q];
    box on; xlim([xx(1),xx(end)])
    
    
    if Ltestid==1
        title('Deviance')
    end
    
    if Ltestid==NLtest
        xlabel('R')
    end
    
    
    
    linkaxes(pp);linkaxes(qq)
    
    
    
    
end