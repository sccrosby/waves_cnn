% Figure 
clearvars

% Global set
set(0,'defaultaxesfontsize',8)

% Pick 3 models to plot
fol1 = '46211_20190717103623_dev_dev_48-24_2019_07_17_134438'
fol2 = '46211_20190717103623_dev_test_48-24_2019_07_17_134507'
fol3 = '46211_20190717103623_dev_test_48-24_2019_07_17_134507'




if isunix && strfind(pwd, 'jonny')
    fol1 = strcat('/home/hutch_research/projects/ml_waves19_jonny/data_outputs/',fol1);
    fol2 = strcat('/home/hutch_research/projects/ml_waves19_jonny/data_outputs/',fol2);
    fol3 = strcat('/home/hutch_research/projects/ml_waves19_jonny/data_outputs/',fol3);
    addpath /home/hutch_research/projects/ml_waves19/MatlabPlottingCode/cbrewer
end

clf
mycolors = lines(5);
myylim = [20 70];

pleft = .1;
pright = .05;
ptop = .05;
pbot = .1;
pspace = .01;
pwid = (1-pleft-pright-2*pspace)/3;
pheight = 1-ptop-pbot;

%-------------------------------------------------------------------------%
%--------------------- PLOT 1 --------------------------------------------%
%-------------------------------------------------------------------------%
% Load data
files=dir(fol1);
fname = files(4).name(1:17);
disp(fname)
num_hours = 24;
[ obs, ww3, pred] = load_data( fol1, fname, num_hours );
pred = removeOutofBounds( pred, ww3, num_hours, 'all' );

% Next plot show a metric of fitting spectral shape
% Calc %different of spectra shape at each time step and average
abnd = 1:13;
bbnd = 14:28;
abias = repmat(nanmean(ww3.e(:,abnd)-obs.e(:,abnd),2),[1 length(abnd)]);
bbias = repmat(nanmean(ww3.e(:,bbnd)-obs.e(:,bbnd),2),[1 length(bbnd)]);
spdf_ww3_sw = nanmean(sum(abs(ww3.e(:,abnd)-obs.e(:,abnd)-abias),2)./sum(obs.e(:,abnd),2));
spdf_ww3_ss = nanmean(sum(abs(ww3.e(:,bbnd)-obs.e(:,bbnd)-bbias),2)./sum(obs.e(:,bbnd),2));
spdf_pred_sw = NaN(1,num_hours);
spdf_pred_ss = NaN(1,num_hours);
for fh = 1:num_hours
spdf_pred_sw(fh) = nanmean(sum(abs(pred(fh).e(:,abnd)-obs.e(:,abnd)),2)./sum(obs.e(:,abnd),2));
spdf_pred_ss(fh) = nanmean(sum(abs(pred(fh).e(:,bbnd)-obs.e(:,bbnd)),2)./sum(obs.e(:,bbnd),2));
end

axes('position',[pleft+0*(pwid+pspace) pbot pwid pheight])
hold on
lh(1) = plot(1:num_hours,spdf_pred_sw*100,'Color',mycolors(1,:));
plot(1:num_hours,spdf_ww3_sw*ones(1,num_hours)*100,'--','Color',mycolors(1,:))
lh(2) = plot(1:num_hours,spdf_pred_ss*100,'Color',mycolors(2,:));
plot(1:num_hours,spdf_ww3_ss*ones(1,num_hours)*100,'--','Color',mycolors(2,:))
legend(lh,'Swell','Seas')
ylabel('Spectral Misfit [%]')
xlabel('Forecast hour')
grid on
box on
ylim(myylim)


%-------------------------------------------------------------------------%
%--------------------- PLOT 2 --------------------------------------------%
%-------------------------------------------------------------------------%
files=dir(fol2);
fname = files(4).name(1:17);
disp(fname)
num_hours = 24;
[ obs, ww3, pred] = load_data( fol2, fname, num_hours );
pred = removeOutofBounds( pred, ww3, num_hours, 'all' );

abnd = 1:13;
bbnd = 14:28;
abias = repmat(nanmean(ww3.e(:,abnd)-obs.e(:,abnd),2),[1 length(abnd)]);
bbias = repmat(nanmean(ww3.e(:,bbnd)-obs.e(:,bbnd),2),[1 length(bbnd)]);
spdf_ww3_sw = nanmean(sum(abs(ww3.e(:,abnd)-obs.e(:,abnd)-abias),2)./sum(obs.e(:,abnd),2));
spdf_ww3_ss = nanmean(sum(abs(ww3.e(:,bbnd)-obs.e(:,bbnd)-bbias),2)./sum(obs.e(:,bbnd),2));
spdf_pred_sw = NaN(1,num_hours);
spdf_pred_ss = NaN(1,num_hours);
for fh = 1:num_hours
spdf_pred_sw(fh) = nanmean(sum(abs(pred(fh).e(:,abnd)-obs.e(:,abnd)),2)./sum(obs.e(:,abnd),2));
spdf_pred_ss(fh) = nanmean(sum(abs(pred(fh).e(:,bbnd)-obs.e(:,bbnd)),2)./sum(obs.e(:,bbnd),2));
end

axes('position',[pleft+1*(pwid+pspace) pbot pwid pheight])
hold on
lh(1) = plot(1:num_hours,spdf_pred_sw*100,'Color',mycolors(1,:));
plot(1:num_hours,spdf_ww3_sw*ones(1,num_hours)*100,'--','Color',mycolors(1,:))
lh(2) = plot(1:num_hours,spdf_pred_ss*100,'Color',mycolors(2,:));
plot(1:num_hours,spdf_ww3_ss*ones(1,num_hours)*100,'--','Color',mycolors(2,:))
set(gca,'YTickLabel',[])
xlabel('Forecast hour')
grid on
box on
ylim(myylim)

%-------------------------------------------------------------------------%
%--------------------- PLOT 3 --------------------------------------------%
%-------------------------------------------------------------------------%
files=dir(fol3);
fname = files(4).name(1:17);
disp(fname)
num_hours = 24;
[ obs, ww3, pred] = load_data( fol3, fname, num_hours );
pred = removeOutofBounds( pred, ww3, num_hours, 'all' );

abnd = 1:13;
bbnd = 14:28;
abias = repmat(nanmean(ww3.e(:,abnd)-obs.e(:,abnd),2),[1 length(abnd)]);
bbias = repmat(nanmean(ww3.e(:,bbnd)-obs.e(:,bbnd),2),[1 length(bbnd)]);
spdf_ww3_sw = nanmean(sum(abs(ww3.e(:,abnd)-obs.e(:,abnd)-abias),2)./sum(obs.e(:,abnd),2));
spdf_ww3_ss = nanmean(sum(abs(ww3.e(:,bbnd)-obs.e(:,bbnd)-bbias),2)./sum(obs.e(:,bbnd),2));
spdf_pred_sw = NaN(1,num_hours);
spdf_pred_ss = NaN(1,num_hours);
for fh = 1:num_hours
spdf_pred_sw(fh) = nanmean(sum(abs(pred(fh).e(:,abnd)-obs.e(:,abnd)),2)./sum(obs.e(:,abnd),2));
spdf_pred_ss(fh) = nanmean(sum(abs(pred(fh).e(:,bbnd)-obs.e(:,bbnd)),2)./sum(obs.e(:,bbnd),2));
end

axes('position',[pleft+2*(pwid+pspace) pbot pwid pheight])
hold on
lh(1) = plot(1:num_hours,spdf_pred_sw*100,'Color',mycolors(1,:));
plot(1:num_hours,spdf_ww3_sw*ones(1,num_hours)*100,'--','Color',mycolors(1,:))
lh(2) = plot(1:num_hours,spdf_pred_ss*100,'Color',mycolors(2,:));
plot(1:num_hours,spdf_ww3_ss*ones(1,num_hours)*100,'--','Color',mycolors(2,:))
set(gca,'YTickLabel',[])
xlabel('Forecast hour')
grid on
box on
ylim(myylim)

printFig(gcf,'fig_spec_misfit',[8 4],'png')





