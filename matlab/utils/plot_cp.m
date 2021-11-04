function plot_cp(CPmean, CPstd)

JJ = size(CPmean,2);
II = size(CPmean{1,1},2);


figure
for rr=1:II
    for jj=1:JJ
      subplot(II,JJ,jj+(rr-1)*JJ); hold on
      
      % Current component
      pcur = CPmean{1,jj}(:,rr);
      ta = 1:size(pcur,1);
      
      if (nargin >1) && not(isempty(CPstd))
          % Diagonal of the covariance matrix
          scur = CPstd{1,jj}(:, rr + (rr-1)*II);
          
          % + and - 2 std
          up =  pcur+1*sqrt(abs(scur));
          lo =  pcur-1*sqrt(abs(scur));
          
          % Patch the std intervals
          patch([ta(:); flipud(ta(:))]', [up(:); flipud(lo(:))]',...
              'k', 'FaceAlpha',0.2,'EdgeAlpha',0)
      end
      
      % Plot components
      scatter(ta, pcur, 30,'k', 'filled')
      axis tight; box on
      
      if rr==1
         title(['dim= ', num2str(jj)]) 
      end
      
      if jj==1
          ylabel(['r= ', num2str(rr)])
      end
      
    end 
end

end