addpath('./utils')

%% Preliminaries: Polya-Gamma densities PG(b,c)
% Distribution Parameters
b = [1,10,10,100];
c = [0,0,100,10];

% Estimate densities
[pdf_tot,querry_tot] = pg_psd(b,c);

% True moments of PG(b,c)
true_moments     = pg_moment(b(:),c(:));

% Estimate moments
du = querry_tot(2,:)-querry_tot(1,:);
test_moments = [(sum(pdf_tot.*querry_tot.^1,1).*du)',...
    (sum(pdf_tot.*querry_tot.^2,1).*du)',...
    (sum(pdf_tot.*querry_tot.^3,1).*du)'];

figure
for param_id = 1:length(b)
    subplot(1,length(b),param_id)
    plot(querry_tot(:,param_id),pdf_tot(:,param_id), 'linewidth',2,'color','k')
    title(['PG( ', num2str(b(param_id)),' , ',  num2str(c(param_id)) ,' )'])
    
    
    disp(['True Moments:' ,...
        ' E(1) = ', num2str(true_moments(param_id,1)','%11.4g'),...
        ' E(2) = ', num2str(true_moments(param_id,2)','%11.4g'),...
        ' E(3) = ', num2str(true_moments(param_id,3)','%11.4g')])
    
    disp(['Test Moments:' ,...
        ' E(1) = ', num2str(test_moments(param_id,1)','%11.4g'),...
        ' E(2) = ', num2str(test_moments(param_id,2)','%11.4g'),...
        ' E(3) = ', num2str(test_moments(param_id,3)','%11.4g')])
    
    disp(' ')
end
set(gcf,'position', [633 1589 1148 230])
box on


%% Example KL divergence between PG densities

b1 = [200,200];  c1 = [0,0];
b2 = [50,80];  c2 = [0,0];

[pdf1,querry1] = pg_psd(b1,c1);
[pdf2,querry2] = pg_psd(b2,c2);

% Consider using pg_kl here..
[pdf1_bis, pdf2_bis, querry] = pg_pdf_align(pdf1,pdf2,querry1,querry2);
KL  = pg_kl(pdf1,pdf2,querry1,querry2);


uquerry = 1e-5:0.01:4*max(b1);
[alpha,beta] = pg_to_ig(b1,c1);
pdf_pg_ig1 = ig_pdf(uquerry,alpha,beta);

[alpha,beta] = pg_to_ig(b2,c2);
pdf_pg_ig2 = ig_pdf(uquerry,alpha,beta);

figure
for ll = 1:size(pdf1,2)
subplot(1,size(pdf1,2),ll)
hold on 
plot(querry1(:,ll),pdf1(:,ll),'linewidth',3,'color',[0,0,0.8],'linestyle', '-')
plot(querry2(:,ll),pdf2(:,ll),'linewidth',3,'color',[0.8,0,0],'linestyle', '-')

plot(querry(:,ll),pdf1_bis(:,ll),'linewidth',3,'color',[0,1,0.8],'linestyle', '--')
plot(querry(:,ll),pdf2_bis(:,ll),'linewidth',3,'color',[0.8,1,0],'linestyle', '--')

box on 

name1= ['', num2str(b1(ll),2),',', num2str(c1(ll),2) ,''];
name2= ['', num2str(b2(ll),2),',', num2str(c2(ll),2) ,''];

title(['KL(',name1,'|',name2,')=' , num2str(KL(ll))])

end

legend('PG1','PG2','Joint Support 1','Joint Support 2')
set(gcf,'position', [373        1042        1413         432])


%% Numerical KL divergence between PG(b,c) using a range of value and Inverse Gamma Moment Matching
clear

% True Params
btrue = 50;
ctrue = 1;

% Gridseach Size
num_param_b = 100;
num_param_c = 50;

% Gridsearch
btest = linspace(btrue*1/5, 150,num_param_b)';
ctest = linspace(ctrue*1/9, 6,num_param_c)';

[Btest,Ctest] = meshgrid(btest,ctest);
[Btrue,Ctrue] = meshgrid(btrue*ones(num_param_b,1),ctrue*ones(num_param_c,1));

% Estimate PG densities
disp('Estimating PG densities...')
[pdf_true,querry_true] = pg_psd(Btrue(:),Ctrue(:),150,12);
[pdf_test,querry_test] = pg_psd(Btest(:),Ctest(:),150,12);
disp('Estimating PG densities... done')

% Numerical KL divergence
disp('Numerical KL divergence...')
KL_pg_numerical = pg_kl(pdf_true,pdf_test,querry_true,querry_test);
KL_pg_numerical = reshape(KL_pg_numerical,num_param_c,num_param_b)';
disp('Numerical KL divergence... done')

disp('Moment Match KL divergence...')
% Moment Matching With Inverse Gamma Distribution
KL_pg_mm = pg_kl_moment_match(Btrue(:),Btest(:),Ctrue(:),Ctest(:),'MM-G');
KL_pg_mm = reshape(KL_pg_mm,num_param_c,num_param_b)';
[minp, minloc_mm] = min(KL_pg_mm(:));
disp('Moment Match KL divergence... done')

% Index of interest
[~, bid_single] = min(abs(btrue-btest));
[~, cid_single] = min(abs(ctrue-ctest));
[minp, minloc_num] = min(KL_pg_numerical(:));
Btest2 = Btest';
Ctest2 = Ctest';


%%
% Color Limites
ylim0 = [0,30];

figure; 
% 2D KL Numerical
subplot(2,3,1)
imagesc(ctest,btest,(KL_pg_numerical)); hold on
colormap(pink)
colorbar
line([min(ctest),max(ctest)], [btest(bid_single), btest(bid_single)], 'color', 'm','LineWidth',2)
line([ctest(cid_single),ctest(cid_single)], [min(btest),max(btest)], 'color', 'm','LineWidth',2)
p1=scatter(ctrue,btrue, 400, 'm','+');
p2=scatter(Ctest2(minloc_num),Btest2(minloc_num), 400, 'g', '+','LineWidth',5);
legend([p1,p2],{'True','KL fit'})
title('PG : KL(b_0,c_0 || b,c)')
caxis(ylim0) ;xlabel('c');ylabel('b')

% 1D KL(b,:) Numerical
subplot(2,3,2);hold on
plot(ctest, KL_pg_numerical(bid_single,:) , 'color', 'k','LineWidth',2)
scatter(ctest(cid_single), KL_pg_numerical(bid_single,cid_single), 400, 'm', '+','LineWidth',5);
axis tight; box on ;xlabel('c') ;xlim([ctest(1),ctest(end)])
title('PG : KL(b_0,c_0 || b_0,c)')
ylim(ylim0)

% 1D KL(:,c) Numerical
subplot(2,3,3); hold on
plot(btest, KL_pg_numerical(:,cid_single) , 'color', 'k','LineWidth',2)
scatter(btest(bid_single), KL_pg_numerical(bid_single,cid_single), 400, 'm', '+','LineWidth',5);
axis tight; box on ;xlabel('b') ;xlim([btest(1),btest(end)])
title('PG : KL(b_0,c_0 || b,c_0)')
ylim(ylim0)

% 2D KL Moment Match
subplot(2,3,4)
imagesc(ctest,btest,KL_pg_mm);hold on
colormap(pink)
colorbar
line([min(ctest),max(ctest)], [btest(bid_single), btest(bid_single)], 'color', 'm','LineWidth',2)
line([ctest(cid_single),ctest(cid_single)], [min(btest),max(btest)], 'color', 'm','LineWidth',2)
p1=scatter(ctrue,btrue, 400, 'm','+');
p2=scatter(Ctest2(minloc_mm),Btest2(minloc_mm), 400, 'g', '+','LineWidth',5);
colormap(pink);colorbar
xlabel('c');ylabel('b');hold on
title('MM-G : KL(b_0,c_0 || b,c)')
caxis([0 35])

% 1D KL(b,:) MM
subplot(2,3,5); hold on
plot(ctest, KL_pg_mm(bid_single,:) , 'color', 'k','LineWidth',2)
scatter(ctest(cid_single), KL_pg_mm(bid_single,cid_single), 400, 'm', '+','LineWidth',5);
axis tight; box on ;xlabel('c') ;xlim([ctest(1),ctest(end)])
title('MM-G : KL(b_0,c_0 || b_0,c)')
ylim(ylim0)

% 1D KL(:,c) MM
subplot(2,3,6); hold on
plot(btest, KL_pg_mm(:,cid_single) , 'color', 'k','LineWidth',2)

scatter(btest(bid_single), KL_pg_mm(bid_single,cid_single), 400, 'm', '+','LineWidth',5);
axis tight; box on ;xlabel('b') ;xlim([btest(1),btest(end)])
title('MM-G : KL(b_0,c_0 || b,c_0)')
ylim(ylim0)

set(gcf,'position', [405 913 1376 561])

%% Validity of the approx. on characteristic functions: Characteristic Functions examples

% Querries
t = linspace(-100,100,10000);

% Example Params
btot = [40,40,40];
ctot = [0,6,20];

xlimtot = [[-5,5]; [-20,20]; [-50,50]];

figure; 
for bid = 1:length(btot)

% Moment Match
b = btot(bid);
c = ctot(bid);

moments = pg_moment(b,c);
pgm = moments(1);
pgv = moments(2)-moments(1).^2;

% Inverse Gamma
alpha = (4+3*b)/2;
beta  = b*(2+3*b)/8;
Fap = besselk(alpha, sqrt(-4*1i*beta*t)).*2.*((-1i*beta*t).^(0.5*alpha))/gamma(alpha);

% Gamma
alpha = (b/4).^2./(b./24);
beta  = 24/4;

alpha = pgm.^2/pgv;
beta  = pgm/pgv;
Fap = (1-1i*t./beta).^(-alpha);

% Characteristic Function
Fpg =  (cosh(0.5*c)./cosh(sqrt((0.5*c.^2-1i*t)/2))).^(b);


d = sum(abs(Fap-Fpg).^2)*mean(diff(t));
n1 = sum(abs(Fap).^2)*mean(diff(t));
n2 = sum(abs(Fpg).^2)*mean(diff(t));
d = 2*d/(n1+n2);
disp(d)

subplot(length(btot),2,1+2*(bid-1));hold on
plot(t, real(Fpg),'color', 'k','LineWidth',2);
plot(t, real(Fap),'color', 'm','LineWidth',2,'linestyle','-.');
title(['Re(f) - PG(' , num2str(b), ',', num2str(c) ,')'])
box on
xlim(xlimtot(bid,:))

if bid ==1
   legend('PG','IG') 
end

subplot(length(btot),2,2+2*(bid-1));hold on
plot(t, imag(Fpg),'color', 'k','LineWidth',2);
plot(t, imag(Fap),'color', 'm','LineWidth',2,'linestyle','-.');
title(['Re(f) - PG(' , num2str(b), ',', num2str(c) ,')'])
box on
xlim(xlimtot(bid,:))

end
set(gcf,'position', [701   976   783   695])


%% Validity of the approx. on characteristic functions: compare gridsearch 

% Number of test
num_param_b = 80;
num_param_c = 60;

btest_char = linspace(2,100,num_param_b)';

fano_factor = linspace(1+1/num_param_c,2,num_param_c)';
ctest_char = linspace(0,5 ,num_param_c)';

ctest_char= -log(fano_factor-1);

% Querry for characteristic functions
t = linspace(-10,10,10000);

% Moment Match
[Btest_char,Ctest_char] = meshgrid(btest_char,ctest_char);

[alpha_ig,beta_ig] = pg_to_ig(Btest_char(:),Ctest_char(:));
[alpha_g,beta_g]   = pg_to_g(Btest_char(:),Ctest_char(:));

% IG characteristic Function
%fIG = besselk(alpha_ig.*ones(1,length(t)),...
%    sqrt(-4*1i*beta_ig*t)).*2.*((-1i*beta_ig*t).^(0.5*alpha_ig))./gamma(alpha_ig);
z = sqrt(-4*1i*t).*sqrt(beta_ig(:)./(alpha_ig(:).^2)); z21 = sqrt(1+z.^2);
fIG = exp(alpha_ig.*(1- z21  +log(0.5+0.5*z21)))./sqrt(z21);

% Gamma characteristic function
fG = (1-1i*t./beta_g).^(-alpha_g);

% Normal characteristic Function 
mtot = pg_moment(Btest_char(:),Ctest_char(:));
means = mtot(:,1); varia = mtot(:,2) - mtot(:,1).^2;
fN = exp(1i*means*t-0.5*varia.*t.^2);

% PG characteristic Function
fPG = (cosh(0.5*Ctest_char(:))./cosh(sqrt(0.5*(0.5*Ctest_char(:).^2-1i*t)))).^(Btest_char(:));

% Distances tmp
dt = mean(diff(t));
distance_PI = sum(abs((fPG-fIG).^2)*dt,2);
distance_PG = sum(abs((fPG-fG).^2)*dt,2);
distance_PN = sum(abs((fPG-fN).^2)*dt,2);

% Norms
norm_P  = sum(abs((fPG).^2)*dt,2);
norm_IG = sum(abs((fIG).^2)*dt,2);
norm_G  = sum(abs((fG).^2)*dt,2);
norm_N  = sum(abs((fN).^2)*dt,2);

% Distances
distance_gridsearch_pg_ig = 2*distance_PI./(norm_P+norm_IG);
distance_gridsearch_pg_ig = reshape(distance_gridsearch_pg_ig, [num_param_c,num_param_b])';

distance_gridsearch_pg_g = 2*distance_PG./(norm_P+norm_G);
distance_gridsearch_pg_g = reshape(distance_gridsearch_pg_g, [num_param_c,num_param_b])';

distance_gridsearch_pg_n = 2*distance_PN./(norm_P+norm_N);
distance_gridsearch_pg_n = reshape(distance_gridsearch_pg_n, [num_param_c,num_param_b])';

% Theoretical prediction
avg_spike_count = floor(linspace(1,20,6));
xi_n = fano_factor.*(1+avg_spike_count)./(fano_factor-1);

%% Plots
climits = log([0.00015 0.005]);
climits = [-9,-5];

figure;
% PG - Gamma
subplot(1,3,1)
imagesc(fano_factor, btest_char,log(distance_gridsearch_pg_g)); colorbar
xlabel('FF');ylabel('\xi');colormap(pink)
hold on;
color_bn = [...
    linspace(0.5,0.7,length(avg_spike_count))',...
    linspace(0.5,0.7,length(avg_spike_count))',...
    linspace(0.5,0.7,length(avg_spike_count))'];

for nn =1:length(avg_spike_count)
    plot(fano_factor,xi_n(:,nn),'linewidth',1 , 'color', color_bn(nn,:),'linestyle','-')
end
caxis(climits)
title('d(f_{PG},f_{G})')
xticks([1, 1.25, 1.5,1.75,2]);xlim([1-0.0001,max(fano_factor)])

% PG - Inverse Gamma
subplot(1,3,2)
imagesc(fano_factor, btest_char,log(distance_gridsearch_pg_ig)); colorbar
xlabel('FF');ylabel('\xi');colormap(pink)
hold on;
for nn =1:length(avg_spike_count)
    plot(fano_factor,xi_n(:,nn),'linewidth',1 , 'color', color_bn(nn,:),'linestyle','-')
end
caxis(climits)
title('d(f_{PG},f_{IG})')
xticks([1, 1.25, 1.5,1.75,2]);xlim([1-0.0001,max(fano_factor)])

% PG - Normal
subplot(1,3,3)
imagesc(fano_factor, btest_char,log(distance_gridsearch_pg_n)); colorbar
xlabel('FF');ylabel('\xi');colormap(pink)
hold on;
for nn =1:length(avg_spike_count)
    plot(fano_factor,xi_n(:,nn),'linewidth',1 , 'color', color_bn(nn,:),'linestyle','-')
end
caxis(climits)
title('d(f_{PG},f_{N})')
legend(cellstr(num2str(avg_spike_count')), 'location', 'southeast')
xticks([1, 1.25, 1.5,1.75,2]);xlim([1-0.0001,max(fano_factor)])

set(gcf,'position', [1921         675        1679         287])

%% Paper Figure

climits = [-9,-5];
color_bn = [...
    linspace(0.4,0.9,length(avg_spike_count))',...
    linspace(0.4,0.9,length(avg_spike_count))',...
    linspace(0.4,0.9,length(avg_spike_count))'];


figure; 
% 2D KL Numerical
subplot(1,4,1)
imagesc(ctest,btest,(KL_pg_numerical)); hold on
colormap(pink)
colorbar
p1=scatter(ctrue,btrue, 400, 'g', '+','LineWidth',3);
p2=scatter(Ctest2(minloc_num),Btest2(minloc_num), 200, 'm', 'x','LineWidth',3);
%title('PG : KL(b_0,c_0 || b,c)')
title(['PG : KL(', num2str(btrue),',',num2str(ctrue), '|| \xi, \omega)'])
%legend('PG(b_0, c_0)', 'location', 'southwest')
legend([p1,p2], {'True','KLmin'}, 'location', 'southeast')
caxis(ylim0) ;xlabel('\omega');ylabel('\xi')

% 2D KL Moment Match
subplot(1,4,2)
imagesc(ctest,btest,KL_pg_mm);hold on
colormap(pink)
colorbar
p1=scatter(ctrue,btrue, 400, 'g','+', '+','LineWidth',3);
p2=scatter(Ctest2(minloc_mm),Btest2(minloc_mm), 200, 'm', 'x','LineWidth',3);
colormap(pink);colorbar
xlabel('\omega');ylabel('\xi');hold on
title(['MM-G : KL(', num2str(btrue),',',num2str(ctrue), '|| \xi, \omega)'])
caxis(ylim0)

% Characteristic function distance with theoretical prediction
subplot(1,4,3)
imagesc(fano_factor, btest_char,log(distance_gridsearch_pg_g)); colorbar
xlabel('FF');ylabel('\xi');colormap(pink)
hold on;
for nn =1:length(avg_spike_count)
    plot(fano_factor,xi_n(:,nn),'linewidth',1.5 , 'color', color_bn(nn,:),'linestyle','-')
end
caxis(climits)
title('log d(f_{PG},f_{G})')
xticks([1, 1.25, 1.5,1.75,2]);xlim([1-0.0001,max(fano_factor)])


subplot(1,4,4);
for nn =1:length(avg_spike_count)
    semilogy(fano_factor,error_bound(nn,:),...
        'linewidth',2 , 'color', color_bn(nn,:),'linestyle','-')
    hold on;
end
box on
box on; %axis tight
xlabel('FF')
title('d(f_{PG},f_{G})')
grid on
colorbar

set(gcf,'position', [5 1455 1907 273])
legend(cellstr(num2str(avg_spike_count')), 'location', 'southeast')

%% Load a pre-built data base of PG(b,c) samples (see end)

clear
load('./../pg_tables/pg_psd_tot.mat')
load('./../pg_tables/pg_samples.mat')

%% Compare: Analytic Estimates Vs. (Samples & Moment Matcher)
% Moment matcher: - Generalized Gamma : Fails
%                 - Inverse Gamma     : Fails for b small
% Samples :       - Samples + Kernel  : Fails for c > 0  

% PG(b,c) parameters
btest_tot = [1,10,50,100];
ctest_tot = [0,0,100,100];

btest_tot = [2,10,50,100];
ctest_tot = [4,4,4,4];

% Color Plots
cyan        = [0.2 0.8 0.8];
brown       = [0.2 0 0];
orange      = [1 0.5 0];
blue        = [0 0.5 1];
green       = [0 0.6 0.3];
red         = [1 0.2 0.2];
black       = [0,0,0];

% Analytic Estimates
[pdf_tot,querry_tot] = pg_psd(btest_tot(:)',0*ctest_tot);

figure
for bid = 1:length(btest_tot)
disp([num2str(bid),'/', num2str(length(btest_tot))])
    
btest = btest_tot(bid);
ctest = 0;
du = 0.001; uquerry = 1e-5:du:4*btest;

% Obtain Samples
samples_b0 = get_samples(btest,pg_psd_tot.pg_param,pg_samples.samples_01,pg_samples.samples_n);

% Deduce density using kernels
psd_kernel = ksdensity(samples_b0,uquerry);

% Deduce Density with moment matching generalized gamma
[a,d,p]  = pg_to_gg(btest,ctest);
pdf_pg_gg = gg_pdf(uquerry,a,d,p);

% Deduce Density with moment matching inversed gamma
[alpha,beta] = pg_to_ig(btest,ctest);
pdf_pg_ig1 = ig_pdf(uquerry,alpha,beta);

% Deduce Density with moment matching gamma
[alpha_g,beta_g] = pg_to_g(btest,ctest);
pdf_pg_g1 = g_pdf(uquerry,alpha_g,beta_g);

% Deduce Density with moment matching Normal
[mm] = pg_moment(btest,ctest);
mean = mm(:,1);
vari = mm(:,2)-mm(:,1).^2;
pdfn = exp(-0.5*(uquerry-mean).^2./vari)./sqrt(2*pi*vari);

subplot(2,length(btest_tot),bid+0*length(btest_tot)); hold on
% Samples
histogram(samples_b0, 'Normalization' , 'pdf', 'facecolor', black);
xl = xlim;
% Analytic + Conv Estimate
plot(querry_tot(:,bid),pdf_tot(:,bid), 'color',black, 'linewidth', 2.5)
% Generalized Gamma
plot(uquerry,pdf_pg_gg  , 'color',green, 'linewidth', 2.5)
% Normal Distribution
plot(uquerry,pdfn  , 'color',red, 'linewidth', 2.5)
xlim(xl); box on
if bid==1
   legend({'Samples','Numerical','MM-GenGamma','MM-Gaussian'}) 
end

title(['PG( ', num2str(btest),' , ', num2str(ctest) ,' )'])

subplot(2,length(btest_tot),bid+1*length(btest_tot)); hold on
% Samples
histogram(samples_b0, 'Normalization' , 'pdf', 'facecolor', black);
xl = xlim;
% Analytic + Conv Estimate
plot(querry_tot(:,bid),pdf_tot(:,bid), 'color',black, 'linewidth', 2.5)
% Inverse Gamma
p1=plot(uquerry,pdf_pg_ig1, 'color',orange, 'linewidth', 2.5, 'linestyle','-');
xlim(xl)
% Gamma
p2=plot(uquerry,pdf_pg_g1, 'color',blue, 'linewidth', 2.5, 'linestyle','-');
xlim(xl)

if bid==1
   legend([p1,p2],{'MM-InvGamma','MM-Gamma'})
end

box on
end

set(gcf,'position', [1921 390 1902 560])

%% Helpers: Polya-Gamma Sampler And Moment-Matcher

function [alpha,beta] = pg_to_g(b,c)
% Fit Gamma to PG(b,c) using moment matching

% Moment of the PG distribution
mtot = pg_moment(b,c);

m1 = mtot(:,1);
m2 = mtot(:,2);

m = m1;
v = m2-m1.^2;

alpha = m.^2./v;
beta  = m./v;

end


function pdf = ig_pdf(x,alpha,beta)
    logp = -gammaln(alpha(:)) + alpha(:).*log(beta(:)) - (alpha(:)+1)*log(x(:)') - beta(:)./x(:)' ;
    pdf = exp(logp);
end

function pdf = g_pdf(x,alpha,beta)
    logp = -gammaln(alpha(:)) + alpha(:).*log(beta(:)) + (alpha(:)-1)*log(x(:)') - x(:)'.*beta(:) ;
    pdf = exp(logp);
end



function KL = ig_kl(alpha1,alpha2,beta1,beta2)

alpha1 = alpha1(:);
alpha2 = alpha2(:);

beta1 = beta1(:);
beta2 = beta2(:);

k1 = (alpha1-alpha2).*psi(alpha1);
k2 = (gammaln(alpha2)-gammaln(alpha1));
k3 = alpha2.*(log(beta1)-log(beta2));
k4 = alpha1.*(beta2-beta1)./beta1;


KL = k1 + k2 + k3 + k4;

end

function [alpha,beta] = pg_to_ig(b,c)
% Fit Inverse Gamma to PG(b,c) using moment matching

% Moment of the PG distribution
mtot = pg_moment(b,c);

m1 = mtot(:,1);
m2 = mtot(:,2);

alpha = (2*m2-m1.*m1)./(m2-m1.*m1);
beta  = (m1.*m2)./(m2-m1.*m1);

end


function [pdf12,loc12] = conv_disjoint(pdf1,pdf2,loc1,loc2)
% Convolution between two signals with disjoint support
% Outputs has the same size as inputs

n = length(loc1);

% Sampling rates
d1 = mean(diff(loc1));
d2 = mean(diff(loc2));

loctot = [loc1;loc2];
pdftot = [pdf1;pdf2];

% Sort by Sampling rates
[~,tointerp] = sort([d1,d2], 'descend');
loctot = loctot(tointerp,:);
pdftot = pdftot(tointerp,:);

% Use interpolation on the bigger rate
loc_interp = loctot(1,:);
pdf_interp = pdftot(1,:);

loc2 = loctot(2,:);
pdf2 = pdftot(2,:);

% interpolate
dx = min([d1,d2]);
loc1 = loc_interp(1):dx:loc_interp(end);

pdf1 = interp1(loc_interp,pdf_interp,loc1);

% Convolve
pdf_tmp = conv(pdf1,pdf2)*dx;
loc_tmp = (loc1(1)+loc2(1)):dx:(loc1(end)+loc2(end));

% Downsample back
loc12 = linspace(loc_tmp(1),loc_tmp(end),n);
pdf12 = interp1(loc_tmp,pdf_tmp,loc12);

end

function [pdf, loc] = get_pdf_from_densities(x,param_01_tot,densities_01,densities_n, locations_01,locations_n)
% Get pdf from PG(x,0) evaluated at loc

% Decompose x as real and integer
param_int = floor(x);
param_01 = x-param_int;

% Get closest pg_param from pg_param_tot
[~,param_loc] = min(abs(param_01-param_01_tot));

% densities for (0,1) part
pdf1 = densities_01(param_loc,:);
loc1 = locations_01(param_loc,:);

if param_int>0
    % densities for integer part
    pdf2 = densities_n(param_int,:);
    loc2 = locations_n(param_int,:);
    
    % Efficient Convolution
    [pdf,loc] = conv_disjoint(pdf1,pdf2,loc1,loc2);
    
else
    % No integer part
    pdf = pdf1;
    loc = loc1;
    
end

end

function [densities,locations] = psd_from_samples(samples,density_estimates)
% Use kernel smoothing to estimate pdf from samples

densities = zeros(size(samples,1), density_estimates);
locations = zeros(size(samples,1), density_estimates);

for ii = 1:size(samples,1)
    disp([num2str(ii), '/', num2str(size(samples,1))])
    samples_cur = samples(ii,:);
    
    [psd_cur,loc_cur] = ksdensity(samples_cur, 'NumPoints',density_estimates);
    
    densities(ii,:) = psd_cur/(sum(psd_cur)*mean(diff(loc_cur)));
    locations(ii,:) = loc_cur;
end

end

function pdf_uke = get_pg_pdf(ktot,shape,pg_psd_tot)
pg_param = pg_psd_tot.pg_param;
uquerry  = pg_psd_tot.uquerry;
psd_01   = pg_psd_tot.psd_01;
psd_n    = pg_psd_tot.psd_n;


pdf_uke = zeros(length(uquerry),length(ktot));

for kid = 1:length(ktot)
    kcur = ktot(kid);

    pdf_uke(:,kid) = pg_psd_old(kcur+shape,pg_param,uquerry,psd_01,psd_n);
    
end

end

function [pdf_pg_x,uquerry] = pg_psd_old(x,pg_param,uquerry,psd_01,psd_n)
% Estimate density of PG(x,0) using densities in pg_psd_tot

% Decompose in integer and real in [0,1]
xint = floor(x);
xrea = x - xint;

% Pameters from 'database'
du = mean(diff(uquerry));

% Check x within 'database' bounds
pg_param_min = min(pg_param);
pg_param_max = size(psd_n,1);
if (x<pg_param_min)
    warning('PG Parameter too small. Estimate might be inacurate');
elseif(x >pg_param_max)
    error('PG Parameter out of range');
end

% p0 density of PG(xrea,0)
[~,param_loc] = min(abs(xrea-pg_param));
psd_01_cur = psd_01(param_loc,:);

if xint > 0
    % pn density of PG(xint,0)
    psd_n_cur  = psd_n(xint,:);
    
    % Use convulotion px = p0 * pn
    %pdf_pg_x = conv(psd_01_cur,psd_n_cur)*du;   
    pdf_pg_x = ifft2(fft2(psd_01_cur).*fft2(psd_n_cur))*du;
    pdf_pg_x = pdf_pg_x(1:length(uquerry));
    
else
    pdf_pg_x = psd_01_cur;
end

end

function wx = get_samples(pg_param,pg_param_tot,pg_samples_01,pg_samples_n)
% Evaluate Polya-Gamma PG(pg_param,0) density at u_querry
% Using PG samples pg_samples_01 and pg_samples_n

param_int = floor(pg_param);
param_01 = pg_param-param_int;

% Get closest pg_param from pg_param_tot
[~,param_loc] = min(abs(param_01-pg_param_tot));


w1 = pg_samples_01(param_loc,:);
if param_int>0
    w2 = pg_samples_n(param_int,:);
else
    w2=0;
end
wx = w1+w2;


end

function pdf = gg_pdf(x,a,d,p)
    logp = -gammaln(d/p) + log(p) - d*log(a) + (d-1)*log(x) - (x/a).^p;
    pdf = exp(logp);
end


function [a,d,p] = pg_to_gg(b,c)
% Fit Generalized Gamma to PG(b,c) using moment matching

% Moment of the PG distribution
mtot = num2cell(pg_moment(b,c));

% Fit moment of GG
[a,d,p] = moment_match_gg(mtot{:});

end


function [a,d,p] = moment_match_gg(m1,m2,m3)
% Fit moment m1,m2,m3 to generalized Gamma)
% Outputs generalized gamma parameters a,d,p
% mi = a^i * gamma(d+i/p)/gamma(d/p

% Reduced system
x = log(m1*m3/(m2*m2));
y = log(m1*m1/m2);

% Solve f(X) = [x,y] using Newton's method
f = @(X) [gammaln(X(2))+gammaln(3*X(2)-2*X(1))-2*gammaln(X(1));
    2*gammaln(X(2))-gammaln(2*X(2)-X(1))-gammaln(X(1))];

J = @(X) [[ -2*psi(3*X(2)-2*X(1))-2*psi(X(1)), psi(X(2))+3*psi(3*X(2)-2*X(1))];
    [psi(2*X(2)-X(1))-psi(X(1)), 2*psi(X(2))-2*psi(2*X(2)-X(1))]];

% Init
uv = [1;2];
loss = 1;

for ite = 1:600
    Fcur = f(uv);
    Jcur = J(uv);
    
    % Newton's step
    uv = uv - 0.1*inv(Jcur+0.0001*eye(2))*(Fcur-[x;y]);
    lnew = (Fcur(1)-x).^2+(Fcur(2)-y).^2;
    dl = lnew-loss;
    
    % Converged ?
    if abs(dl) <eps*loss
        break
    else
        loss = lnew;
    end
end

% Grasp d and p
d = uv(1)/(uv(2)-uv(1));
p = 1/(uv(2)-uv(1));

% Deduce a
a1 = m1*exp(gammaln(uv(1))-gammaln(uv(2)));
a2 = (m2*exp(gammaln(uv(1))-gammaln(2*uv(2)-uv(1))))^(1/2);
a3 = (m3*exp(gammaln(uv(1))-gammaln(3*uv(2)-2*uv(1))))^(1/3);

% Sanity check
if max([abs(a1-a2),abs(a1-a3),abs(a3-a2)])/max([a1,a2,a3]) > 0.1
    warning('Generalized Gamma fit likely failed...')
end

a = mean([a1,a2,a3]);

end


function psd = get_psd(pg_param,pg_param_tot,pg_samples_01,pg_samples_n,u_querry)
% Evaluate Polya-Gamma PG(pg_param,0) density at u_querry
% Using PG samples pg_samples_01 and pg_samples_n

param_int = floor(pg_param);
param_01 = pg_param-param_int;

% Get closest pg_param from pg_param_tot
[~,param_loc] = min(abs(param_01-pg_param_tot));

w1 = pg_samples_01(param_loc,:);
if param_int>0
    w2 = pg_samples_n(param_int,:);
else
    w2=0;
end
wx = w1+w2;

psd = ksdensity(wx,u_querry);

end

function X = pgdraw_01(b, Kapprox)
% Draw Nsample from Polyà-Gamma distribution PG(a,b)
% Using finite sum-of-gammas approximation

%assert(all(a(:)<1), 'Using pgdraw_real yields better precision')
% Generate Kapprox Samples from Gamma(a,1)
gk = randg(repmat(b(:),1,Kapprox));

% Rescale
Gk = gk./(((1:Kapprox)-0.5).^2);

% Approximate infinite sum
X = (1/(2*pi^2)) * sum(Gk,2);

end

function moments = pg_moment(b,c)
% First moments of Polya-Gamma distribution
% For phi log laplace phi = log < exp(-ut)>
% Derive and apply at 0

b = b(:); 
c = c(:)+eps;

% phi'(0)
phi_1 = -b.*(1./(2*c)).*tanh(c/2);

% Limits in c = 0 exist but subject to numerical precision issues.. 
l = 0.01; k = 2;
smoother = @(c,l,k) exp(-1./((c/l).^k));

% phi''(0) and phi'''(0)
P2 = @(c) (1./(4*(cosh(c/2).^2).*c.^3)).*(sinh(c)-c);
P3 = @(c) (1./(4*(cosh(c/2).^2).*c.^5)).*(c.^2.*tanh(c/2) + 3*(c-sinh(c)));

phi_2tmp = @(c) P2(c).*smoother(c,l,k) +1/24.*(1-smoother(c,l,k));
phi_3tmp = @(c) P3(c).*smoother(c,l,k) -1/60.*(1-smoother(c,l,k));

phi_2 = b.*phi_2tmp(c);
phi_3 = b.*phi_3tmp(c);

% Associated Moments
m1 = -phi_1;
m2 = phi_2 + phi_1.^2;
m3 = 2*phi_1.^3 - phi_3 - 3*phi_1.*(phi_2 + phi_1.^2);

moments = [m1,m2,m3];



end


%% Build PG(b,0) samples
% bmin = 0.01;
% bmax = 1;
% Nb = 2000;
% b = linspace(bmin,bmax,Nb);
% Nsamples = 100000;
% pg_samples_01 = zeros(Nb, Nsamples);
% 
% for bid = 1:Nb
%     disp([num2str(bid), '/', num2str(Nb)])
%     bcur = repmat(b(bid), [Nsamples,1]);
%     Xcur = pgdraw_01(bcur, 400);
%     pg_samples_01(bid,:) = Xcur;
% end
% 
% %%
% nmin = 1;
% nmax = 2000;
% ntot = 1:nmax;
% Nsamples = 100000;
% pg_samples_n0 = zeros(nmax, Nsamples);
% 
% for nid = ntot
%     disp([num2str(nid), '/', num2str(nmax)])
%     Xcur = pgdraw(zeros(Nsamples,1)); (c) Copyright Enes Makalic and Daniel F. Schmidt, 2017
%     pg_samples_n0(nid,:) = Xcur;
% end
% pg_samples_n = cumsum(pg_samples_n0,1);

% %% Load pre-built PG samples dataset
% load('./pg_samples.mat')
% 
% %% From samples to density 
% 
% density_estimates = 1000;
% [densities_01,locations_01] = psd_from_samples(pg_samples.samples_01,density_estimates);
% [densities_n, locations_n]  = psd_from_samples(pg_samples.samples_n ,density_estimates);
% 
% 
% %% Save
% pg_densities = struct();
% pg_densities.densities_01 = densities_01;
% pg_densities.locations_01 = locations_01;
% pg_densities.densities_n  = densities_n;
% pg_densities.locations_n  = locations_n;
% pg_densities.param_01 = pg_samples.param_01;
% pg_densities.param_n  = pg_samples.param_n;
% save('pg_densities','pg_densities','-v7.3')


%%
% 
% 
% [pdf_tot,querry_tot] = pg_psd(btest_tot(:)',ctest_tot(:)');
% for bid = 1:length(btest_tot)
% disp([num2str(bid),'/', num2str(length(btest_tot))])
%     
% btest = btest_tot(bid);
% ctest = ctest_tot(bid);
% du = 0.001; uquerry = 1e-5:du:4*btest;
% 
% 
% % Obtain Samples
% samples_b0 = get_samples(btest,pg_psd_tot.pg_param,pg_samples.samples_01,pg_samples.samples_n);
% % Deduce density using kernels
% psd_kernel = ksdensity(samples_b0,uquerry);
% psd_kernel = psd_kernel.*cosh(0.5*ctest).^btest.*exp(-0.5*ctest.^2*uquerry);
% 
% % Deduce Density with moment matching generalized gamma
% [a,d,p]  = pg_to_gg(btest,ctest);
% pdf_pg_gg = gg_pdf(uquerry,a,d,p);
% 
% % Deduce Density with moment matching inversed gamma
% [alpha,beta] = pg_to_ig(btest,ctest);
% pdf_pg_ig1 = ig_pdf(uquerry,alpha,beta);
% 
% % Deduce Density with moment matching gamma
% [alpha_g,beta_g] = pg_to_g(btest,ctest);
% pdf_pg_g1 = g_pdf(uquerry,alpha_g,beta_g);
% 
% subplot(2,length(btest_tot),length(btest_tot)+bid); hold on
% % Samples
% % Fit with kernels
% plot(uquerry,psd_kernel, 'color',black, 'linewidth', 3)
% 
% % Analytic + Conv Estimate
% plot(querry_tot(:,bid),pdf_tot(:,bid), 'color',green, 'linewidth', 3, 'linestyle','--')
% 
% % Generalized Gamma
% plot(uquerry,pdf_pg_gg  , 'color',blue, 'linewidth', 3)
% 
% % Inverse Gamma
% plot(uquerry,pdf_pg_ig1, 'color',red, 'linewidth', 3, 'linestyle','-')
% 
% % Inverse Gamma
% plot(uquerry,pdf_pg_ig1, 'color','m', 'linewidth', 3, 'linestyle',':')
% 
% xlim([querry_tot(1,bid),querry_tot(end,bid)])
% ylim([0, 1.2*max(pdf_tot(:,bid))])
% 
% plot(querry_tot(:,bid),pdf_tot(:,bid), 'color',green, 'linewidth', 3, 'linestyle','--')
% 
% title(['PG( ', num2str(btest),' , ', num2str(ctest) ,' )'])
% box on
% 
% end

