% Figure 
clearvars

% Global set
set(0,'defaultaxesfontsize',8)


% Pick 3 models to plot
fol1 = '46218_20190716195232_dev_dev_48-24_2019_07_17_094516'
fol2 = '46218_20190716195232_dev_test_48-24_2019_07_17_094538'
fol3 = '46218_20190716195232_dev_test_48-24_2019_07_17_094538'

num_hours = 24;
out_of_bounds_flag = 'all';  % 'all', 'some', 'none'


if isunix && strfind(pwd, 'jonny')
    fol1 = strcat('/home/hutch_research/projects/ml_waves19_jonny/data_outputs/',fol1);
    fol2 = strcat('/home/hutch_research/projects/ml_waves19_jonny/data_outputs/',fol2);
    fol3 = strcat('/home/hutch_research/projects/ml_waves19_jonny/data_outputs/',fol3);
    addpath /home/hutch_research/projects/ml_waves19/MatlabPlottingCode/cbrewer
end


clf
mycolors = lines(5);
myylim = [6 20];

pleft = .1;
pright = .05;
ptop = .05;
pbot = .1;
pspace = .01;
pwid = (1-pleft-pright-2*pspace)/3;
pheight = 1-ptop-pbot;

files=dir(fol1);
fname = files(4).name(1:17);
disp(fname)

% Load Data
[ obs, ww3, pred] = load_data( fol1, fname, num_hours );

% Zero out negative energy
pred = zeroNegEvalues( pred, num_hours );

% Remove out of bounds directional moments
pred_orig = removeOutofBounds( pred, ww3, num_hours, 'none' );
pred = removeOutofBounds( pred, ww3, num_hours, out_of_bounds_flag );

% Direction
abnd = 1:13; %0.04-0.105 Hz
obsSW = intMeanDir( obs, 1, abnd );
ww3SW = intMeanDir( ww3, 1, abnd );
predSW = intMeanDir( pred, num_hours, abnd );
predSW_orig = intMeanDir( pred_orig, num_hours, abnd );

% Estimate Hs for SW and SS
ww3.hsSW = 4*sqrt(ww3.e(:,abnd)*ww3.bw(abnd)');
obs.hsSW = 4*sqrt(obs.e(:,abnd)*obs.bw(abnd)');
for hr = 1:num_hours
    pred(hr).hsSW = 4*sqrt(pred(hr).e(:,abnd)*pred(hr).bw(abnd)');
    pred_orig(hr).hsSW = 4*sqrt(pred_orig(hr).e(:,abnd)*pred_orig(hr).bw(abnd)');
end

abnd = 14:28; %0.105 - 0.25 Hz
obsSS = intMeanDir( obs, 1, abnd );
ww3SS = intMeanDir( ww3, 1, abnd );
predSS = intMeanDir( pred, num_hours, abnd );
predSS_orig = intMeanDir( pred_orig, num_hours, abnd );

% Estimate Hs for SW and SS
ww3.hsSS = 4*sqrt(ww3.e(:,abnd)*ww3.bw(abnd)');
obs.hsSS = 4*sqrt(obs.e(:,abnd)*obs.bw(abnd)');
for hr = 1:num_hours
    pred(hr).hsSS = 4*sqrt(pred(hr).e(:,abnd)*pred(hr).bw(abnd)');
    pred_orig(hr).hsSS = 4*sqrt(pred_orig(hr).e(:,abnd)*pred_orig(hr).bw(abnd)');
end

% Estimate RMSE:
diffcut = 90;
temp = ww3SW.md1-obsSW.md1;
temp(temp>diffcut) = NaN;
rmse_ww3_SW = sqrt(nanmean(temp.^2));
for fh = 1:num_hours
    temp = predSW(fh).md1-obsSW.md1;
    temp(temp>diffcut) = NaN;
    rmse_pred_SW(fh) = sqrt(nanmean(temp.^2));
    
    temp_orig = predSW_orig(fh).md1-obsSW.md1;
    temp_orig(temp>diffcut) = NaN;
    rmse_pred_SW_orig(fh) = sqrt(nanmean(temp_orig.^2));
end
temp = ww3SS.md1-obsSS.md1;
temp(temp>diffcut) = NaN;
rmse_ww3_SS = sqrt(nanmean(temp.^2));
for fh = 1:num_hours
    temp = predSS(fh).md1-obsSS.md1;
    temp(temp>diffcut) = NaN;
    rmse_pred_SS(fh) = sqrt(nanmean(temp.^2));
    
    temp = predSS_orig(fh).md1-obsSS.md1;
    temp(temp>diffcut) = NaN;
    rmse_pred_SS_orig(fh) = sqrt(nanmean(temp.^2));
end

% Plot
axes('position',[pleft+0*(pwid+pspace) pbot pwid pheight])
hold on

% Swell plots
lh(1) = plot(1:num_hours,rmse_ww3_SW*ones(1,num_hours),'--','Color',mycolors(1,:));
lh(2) = plot(1:num_hours,rmse_pred_SW,'Color',mycolors(1,:));
lh(3) = plot(1:num_hours,rmse_pred_SW_orig,'-.','Color',mycolors(1,:));

% Seas plots
% plot(1:num_hours,rmse_ww3_SS*ones(1,num_hours),'--','Color',mycolors(2,:))
lh(4) = plot(1:num_hours,rmse_ww3_SS*ones(1,num_hours),'--','Color',mycolors(2,:));
lh(5) = plot(1:num_hours,rmse_pred_SS,'Color',mycolors(2,:));
lh(6) = plot(1:num_hours,rmse_pred_SS_orig,'-.','Color',mycolors(2,:));


legend(lh,'Swell WW3', 'Swell Post', 'Swell Orig', 'Seas WW3', 'Seas Post', 'Seas Orig')
ylabel('\theta_m RMSE [^o]')
xlabel('Forecast hour')
ylim(myylim)
grid on
box on

% Time Series
% fh = 6;
% figure(1)
% clf
% subplot(211)
% hold on
% plot(obsSW.md1)
% plot(ww3SW.md1)
% plot(predSW(fh).md1)
% 
% subplot(212)
% hold on
% plot(obsSS.md1)
% plot(ww3SS.md1)
% plot(predSS(fh).md1)
% return
%


%%%%%%% LOAD AND PROCESS AND PLOT BUOY 2 %%%%%%%%%%%%%%%%%%%%%%
files=dir(fol2);
fname = files(4).name(1:17);

% Load Data
[ obs, ww3, pred] = load_data( fol2, fname, num_hours );

% Zero out negative energy
pred = zeroNegEvalues( pred, num_hours );

% Remove out of bounds directional moments
pred_orig = removeOutofBounds( pred, ww3, num_hours, 'none');
pred = removeOutofBounds( pred, ww3, num_hours, out_of_bounds_flag );

% Direction
abnd = 1:13; %0.04-0.105 Hz
obsSW = intMeanDir( obs, 1, abnd );
ww3SW = intMeanDir( ww3, 1, abnd );
predSW = intMeanDir( pred, num_hours, abnd );
predSW_orig = intMeanDir( pred_orig, num_hours, abnd );

% Estimate Hs for SW and SS
ww3.hsSW = 4*sqrt(ww3.e(:,abnd)*ww3.bw(abnd)');
obs.hsSW = 4*sqrt(obs.e(:,abnd)*obs.bw(abnd)');
for hr = 1:num_hours
    pred(hr).hsSW = 4*sqrt(pred(hr).e(:,abnd)*pred(hr).bw(abnd)');
    pred_orig(hr).hsSW = 4*sqrt(pred_orig(hr).e(:,abnd)*pred_orig(hr).bw(abnd)');
end

abnd = 14:28; %0.105 - 0.25 Hz
obsSS = intMeanDir( obs, 1, abnd );
ww3SS = intMeanDir( ww3, 1, abnd );
predSS = intMeanDir( pred, num_hours, abnd );
predSS_orig = intMeanDir( pred_orig, num_hours, abnd );

% Estimate Hs for SW and SS
ww3.hsSS = 4*sqrt(ww3.e(:,abnd)*ww3.bw(abnd)');
obs.hsSS = 4*sqrt(obs.e(:,abnd)*obs.bw(abnd)');
for hr = 1:num_hours
    pred(hr).hsSS = 4*sqrt(pred(hr).e(:,abnd)*pred(hr).bw(abnd)');
    pred_orig(hr).hsSS = 4*sqrt(pred_orig(hr).e(:,abnd)*pred_orig(hr).bw(abnd)');
end

% Estimate RMSE:
diffcut = 90;
temp = ww3SW.md1-obsSW.md1;
temp(temp>diffcut) = NaN;
rmse_ww3_SW = sqrt(nanmean(temp.^2));
for fh = 1:num_hours
    temp = predSW(fh).md1-obsSW.md1;
    temp(temp>diffcut) = NaN;
    rmse_pred_SW(fh) = sqrt(nanmean(temp.^2));
    
    temp_orig = predSW_orig(fh).md1-obsSW.md1;
    temp_orig(temp>diffcut) = NaN;
    rmse_pred_SW_orig(fh) = sqrt(nanmean(temp_orig.^2));
end
temp = ww3SS.md1-obsSS.md1;
temp(temp>diffcut) = NaN;
rmse_ww3_SS = sqrt(nanmean(temp.^2));
for fh = 1:num_hours
    temp = predSS(fh).md1-obsSS.md1;
    temp(temp>diffcut) = NaN;
    rmse_pred_SS(fh) = sqrt(nanmean(temp.^2));
    
    temp = predSS_orig(fh).md1-obsSS.md1;
    temp(temp>diffcut) = NaN;
    rmse_pred_SS_orig(fh) = sqrt(nanmean(temp.^2));
end


% Plot
axes('position',[pleft+1*(pwid+pspace) pbot pwid pheight])
hold on

% Swell plots
lh(1) = plot(1:num_hours,rmse_ww3_SW*ones(1,num_hours),'--','Color',mycolors(1,:));
lh(2) = plot(1:num_hours,rmse_pred_SW,'Color',mycolors(1,:));
lh(3) = plot(1:num_hours,rmse_pred_SW_orig,'-.','Color',mycolors(1,:));

% Seas plots
% plot(1:num_hours,rmse_ww3_SS*ones(1,num_hours),'--','Color',mycolors(2,:))
lh(4) = plot(1:num_hours,rmse_ww3_SS*ones(1,num_hours),'--','Color',mycolors(2,:));
lh(5) = plot(1:num_hours,rmse_pred_SS,'Color',mycolors(2,:));
lh(6) = plot(1:num_hours,rmse_pred_SS_orig,'-.','Color',mycolors(2,:));

set(gca,'YTickLabel',[])
xlabel('Forecast hour')
ylim(myylim)
grid on
box on

%%%%%%% LOAD AND PROCESS AND PLOT BUOY 3 %%%%%%%%%%%%%%%%%%%%%%
files=dir(fol3);
fname = files(4).name(1:17);

% Load Data
[ obs, ww3, pred] = load_data( fol3, fname, num_hours );

% Zero out negative energy
pred = zeroNegEvalues( pred, num_hours );

% Remove out of bounds directional moments
pred_orig = removeOutofBounds( pred, ww3, num_hours, 'none');
pred = removeOutofBounds( pred, ww3, num_hours, out_of_bounds_flag );

% Direction
abnd = 1:13; %0.04-0.105 Hz
obsSW = intMeanDir( obs, 1, abnd );
ww3SW = intMeanDir( ww3, 1, abnd );
predSW = intMeanDir( pred, num_hours, abnd );
predSW_orig = intMeanDir( pred_orig, num_hours, abnd );

% Estimate Hs for SW and SS
ww3.hsSW = 4*sqrt(ww3.e(:,abnd)*ww3.bw(abnd)');
obs.hsSW = 4*sqrt(obs.e(:,abnd)*obs.bw(abnd)');
for hr = 1:num_hours
    pred(hr).hsSW = 4*sqrt(pred(hr).e(:,abnd)*pred(hr).bw(abnd)');
    pred_orig(hr).hsSW = 4*sqrt(pred_orig(hr).e(:,abnd)*pred_orig(hr).bw(abnd)');
end

abnd = 14:28; %0.105 - 0.25 Hz
obsSS = intMeanDir( obs, 1, abnd );
ww3SS = intMeanDir( ww3, 1, abnd );
predSS = intMeanDir( pred, num_hours, abnd );
predSS_orig = intMeanDir( pred_orig, num_hours, abnd );

% Estimate Hs for SW and SS
ww3.hsSS = 4*sqrt(ww3.e(:,abnd)*ww3.bw(abnd)');
obs.hsSS = 4*sqrt(obs.e(:,abnd)*obs.bw(abnd)');
for hr = 1:num_hours
    pred(hr).hsSS = 4*sqrt(pred(hr).e(:,abnd)*pred(hr).bw(abnd)');
    pred_orig(hr).hsSS = 4*sqrt(pred_orig(hr).e(:,abnd)*pred_orig(hr).bw(abnd)');
end

% Estimate RMSE:
diffcut = 90;
temp = ww3SW.md1-obsSW.md1;
temp(temp>diffcut) = NaN;
rmse_ww3_SW = sqrt(nanmean(temp.^2));
for fh = 1:num_hours
    temp = predSW(fh).md1-obsSW.md1;
    temp(temp>diffcut) = NaN;
    rmse_pred_SW(fh) = sqrt(nanmean(temp.^2));
    
    temp_orig = predSW_orig(fh).md1-obsSW.md1;
    temp_orig(temp>diffcut) = NaN;
    rmse_pred_SW_orig(fh) = sqrt(nanmean(temp_orig.^2));
end
temp = ww3SS.md1-obsSS.md1;
temp(temp>diffcut) = NaN;
rmse_ww3_SS = sqrt(nanmean(temp.^2));
for fh = 1:num_hours
    temp = predSS(fh).md1-obsSS.md1;
    temp(temp>diffcut) = NaN;
    rmse_pred_SS(fh) = sqrt(nanmean(temp.^2));
    
    temp = predSS_orig(fh).md1-obsSS.md1;
    temp(temp>diffcut) = NaN;
    rmse_pred_SS_orig(fh) = sqrt(nanmean(temp.^2));
end


% Plot
axes('position',[pleft+2*(pwid+pspace) pbot pwid pheight])
hold on

% Swell plots
lh(1) = plot(1:num_hours,rmse_ww3_SW*ones(1,num_hours),'--','Color',mycolors(1,:));
lh(2) = plot(1:num_hours,rmse_pred_SW,'Color',mycolors(1,:));
lh(3) = plot(1:num_hours,rmse_pred_SW_orig,'-.','Color',mycolors(1,:));

% Seas plots
% plot(1:num_hours,rmse_ww3_SS*ones(1,num_hours),'--','Color',mycolors(2,:))
lh(4) = plot(1:num_hours,rmse_ww3_SS*ones(1,num_hours),'--','Color',mycolors(2,:));
lh(5) = plot(1:num_hours,rmse_pred_SS,'Color',mycolors(2,:));
lh(6) = plot(1:num_hours,rmse_pred_SS_orig,'-.','Color',mycolors(2,:));



% Plot
% axes('position',[pleft+2*(pwid+pspace) pbot pwid pheight])
% hold on
% lh(1) = plot(1:num_hours,rmse_pred_SW,'Color',mycolors(1,:));
% plot(1:num_hours,rmse_ww3_SW*ones(1,num_hours),'--','Color',mycolors(1,:))
% lh(2) = plot(1:num_hours,rmse_pred_SS,'Color',mycolors(2,:));
% plot(1:num_hours,rmse_ww3_SS*ones(1,num_hours),'--','Color',mycolors(2,:))

set(gca,'YTickLabel',[])
xlabel('Forecast hour')
ylim(myylim)
grid on
box on

printFig(gcf,'fig_dir_rmse',[8 4],'png')


% 
% return
% %%
% tt = 3900;
% fh = 1;
% 
% clf
% plot(obs.fr,obs.e(tt,:))
% hold on
% plot(ww3.fr,ww3.e(tt,:))
% plot(pred(fh).fr,pred(fh).e(tt,:))
% 
% 
% %%
% tt = 3900;
% fh = 1;
% 
% clf
% hold on
% plot(obs.md1SW)
% plot(pred(1).md1SWF,'.')
% hold on
% plot(pred(5).md1SWF)
% 
% 
% %plot(ww3.fr,ww3.e(tt,:))
% %plot(pred(fh).fr,pred(fh).e(tt,:))
% 
% 
% 
% 
% 
% 
% 
