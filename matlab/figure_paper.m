velocity = experimental_parameters.velocity;
position = experimental_parameters.position;
absspeed = abs(velocity);
time_bin = (1:size(velocity))*0.1;

figure
subplot(1,3,1)
plot(time_bin,velocity, 'color','k', 'linewidth',1.2)
box on; xlabel('Sec'); ylabel('deg/sec')
xlim([time_bin(1),time_bin(end)])

subplot(1,3,2)
plot(time_bin,absspeed, 'color','k', 'linewidth',1.2)
box on; xlabel('Sec'); ylabel('deg/sec')
xlim([time_bin(1),time_bin(end)])

subplot(1,3,3)
plot(time_bin,position, 'color','k', 'linewidth',1.2)
box on; xlabel('Sec'); ylabel('deg')
xlim([time_bin(1),time_bin(end)])
set(gcf,'position', [124   528   796   170])

%% 


Rtest = 4;

ref_cur = ref_tot(:,Rtest);

VI_ref = fators_tot_f_ordered{1,Rtest,ref_cur(1)};  VI_ref =  normalize_cp(VI_ref,3);
VI_var_ref = varianc_tot{1,ref_cur(1)}{1,Rtest};

CP_ref = fators_tot_f_ordered{2,Rtest,ref_cur(2)};  CP_ref =  normalize_cp(CP_ref,3);
GCP_ref = fators_tot_f_ordered{3,Rtest,ref_cur(3)}; GCP_ref = normalize_cp(GCP_ref,3);




% Neuron Groups
Pg = experimental_parameters.neuron_group;

corespid = experimental_parameters.neuron_idgrp;
% Plot regions/Layers
region_separators = 0.5+[0;find([diff(corespid(:,2));1])];
region_text = region_separators(1:end-1) + 0.5*(region_separators(2:end)-region_separators(1:end-1));

% Plot Colors
color_condition = [[0,0,1];[0.5,0,0.5];[1,0,0]];
title_condition = {'Vest';'Both';'Visu'};

% Neurons
%color_neurons = colorm(length(unique(corespid)));
%color_neurons = color_neurons(corespid(:,1),:);
color_neurons = [...
    linspace(0.7,0,length(unique(corespid)))',...
    linspace(0.7,0,length(unique(corespid)))',...
    linspace(0.7,0,length(unique(corespid)))'];
color_neurons = color_neurons(corespid(:,1),:);    


[~, group_id] = max(Pg{1},[],2);
[corres_group,group_ordered] = sort(group_id);
group_ordered_color = color_neurons(corres_group,:);

region_separators2 = [1,find(corres_group==4, 1),length(group_id)];

%% Plots
Nmodel = 1;
Rmax = 6;

if Nmodel==1 % VI model
   model_cur_var =  VI_var_ref;
   [model_cur,model_cur_var] = normalize_cp(VI_ref,3,model_cur_var);
elseif Nmodel ==2
   model_cur = normalize_cp(CP_ref,3); 
elseif Nmodel ==3
   model_cur = normalize_cp(GCP_ref,3);  
end

R = size(model_cur{1},2);
D = size(model_cur,2);

% Hand Flipp the sign of some CPs
% !! Signs must multiply to one !
sign_flip = repmat([1,-1,-1], R);
model_cur = flip_model(model_cur,sign_flip);







dict_region = cellstr(experimental_parameters.dict_region);
dict_layers = cellstr(experimental_parameters.dict_layr);

% CP-Factor Labels
title_dimension = {'Neurons','Time Bins','Conditions'};


timelims = (1:70)*0.1;


figure
for rr = 1:Rmax

    % Neurons
    subplot(Rmax,3,3*(rr-1)+1); hold on
    CPneuron = model_cur{1,1}(:,rr);
    % Square sum over regions
    CPneuron = Pg{1,1}'*(CPneuron).^2;        
    % Normalized CP: = mean average Loading
    %CPneuron = (1./sum(Pg{1,1}))'.*CPneuron;
    
    %scatter(1:size(CPneuron,1),CPneuron,60, color_neurons, 'filled');
    %scatter(1:size(CPneuron,1),CPneuron,60, 'k')
    
    CPneuron = model_cur{1,1}(:,rr);
    scatter(1:size(CPneuron,1),CPneuron(group_ordered),20, group_ordered_color, 'filled');
    
    region_separators = region_separators2;
    
    
    % for legend
    pp = [];
    for nn = 1:length(unique(corespid))
        p = scatter(nn,CPneuron(nn),60, color_neurons(nn,:), 'filled');
        pp = [pp,p];
    end
    
    ylimc = ylim;
    % Plot region Separators
    for rsep = 1:length(region_separators)
        line([region_separators(rsep) region_separators(rsep)],[ylimc(1) ylimc(2)],'linewidth',1.5,'color', 'k')
    end
    ylabel(['r=', num2str(rr)])
    box on
    set(gca,'xtick',0.5*(region_separators(1:end-1)+region_separators(2:end)),...
        'XTickLabel',dict_region)
    box on;
    axis tight
    
    if rr == 1
        title(title_dimension{1})
       legend(pp, dict_layers) 
    end
    
    % Temporal Dynamics
    subplot(Rmax,3,3*(rr-1)+2); hold on
    CPtime = model_cur{1,2}(:,rr);
    % Plot std if possible
    if Nmodel==1
        mean = CPtime;
        stdc = sqrt(abs(model_cur_var{1,2}(:,rr + (rr-1)*R)));
        up = mean+stdc;
        lo = mean-stdc;
        % Patch the std intervals
        patch([timelims(:); flipud(timelims(:))]', [up(:); flipud(lo(:))]',...
            'k', 'FaceAlpha',0.2,'EdgeAlpha',0)
    end
    plot(timelims, CPtime,'color','k','linewidth',1.5)
    if rr ==Rmax
       xlabel('Sec') 
    end
    
    box on;
    if rr == 1
        title(title_dimension{2})
    end
    xlim([timelims(1), timelims(end)])
    
    % Experimental condition
    subplot(Rmax,3,3*(rr-1)+3); hold on
    CPtime = model_cur{1,3}(:,rr);
    if Nmodel==1
        t2 = 1:length(CPtime);
        mean = CPtime;
        stdc = sqrt(abs(model_cur_var{1,3}(:,rr + (rr-1)*R)));
        up = mean+stdc;
        lo = mean-stdc;
        % Patch the std intervals
        patch([t2(:); flipud(t2(:))]', [up(:); flipud(lo(:))]',...
            'k', 'FaceAlpha',0.2,'EdgeAlpha',0)
    end
    scatter(1:length(CPtime), CPtime, 70,color_condition,'filled')
    scatter(1:length(CPtime), CPtime, 70,'k')
    set(gca,'xtick',1:length(CPtime),...
        'XTickLabel',title_condition)
    
    box on;
    if rr == 1
        title(title_dimension{3})
    end
    
end

set(gcf,'position', [1361 547 560 420])

if Rmax==R
    set(gcf,'position',[1921 1 633 961])
end

%%

function model_flip = flip_model(model,sign_flip)
% Flip Some CP components
% ! Sign must multiply to one

R = size(model{1},2);
D = size(model,2);

assert(all(prod(sign_flip,2)))
model_flip = model;
for rr=1:R
    for dd=1:D
        model_flip{1,dd}(:,rr) = sign_flip(rr,dd)*model{1,dd}(:,rr);
    end
end
end










