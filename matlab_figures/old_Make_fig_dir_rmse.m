% Figure 
clearvars

% Global set
set(0,'defaultaxesfontsize',8)

% Default data location

fol = '46211_20190601180601_dev_dev_24-24_2019_06_01_191024'; buoy = '46211_dnu24-24_';
% fol = '46211_20190601094513_dev_test_24-24_2019_06_01_101140'; buoy = '46211_dnu24-24_';

% BUOY 46211
% fol = 'paper_46211_20190525103359_dev_predictions'; buoy = '46211_dev_';
% fol = 'paper_46211_20190525103359_test_predictions'; buoy = '46211_test_';
% fol = '46214_20190525165851_dev_dev_24-24_2019_05_29_104239'; buoy = '46211_dnu24-24_';
% fol = '46211_20190524233803_dev_dev_36-24_2019_05_29_105248'; buoy = '46211_dnu36-24_'; 
% fol = '46211_20190524181618_dev_dev_48-24_2019_05_29_105834'; buoy = '46211_dnu48-24_'; 

% BUOY 46214
% fol = 'paper_46214_20190525165851_dev_predictions'; buoy = '46214_dev_';
% fol = 'paper_46214_20190525165851_test_predictions'; buoy = '46214_test_';

% BUOY 46218
% fol = 'paper_46218_20190525201735_dev_predictions'; buoy = '46218_dev_';
% fol = 'paper_46218_20190525201735_test_predictions'; buoy = '46218_test_';


% Shortcuts 
if isunix && strfind(pwd, 'jonny')
    fol = strcat('/home/hutch_research/projects/ml_waves19_jonny/data_outputs/',fol);
%     addpath /home/hutch_research/projects/ml_waves19/MatlabPlottingCode/cbrewer
end

% set the file name based on directory
files=dir(fol);
fname = files(4).name(1:17);

% Load data
num_hours = 24;
[ obs, ww3, pred] = load_data( fol, fname, num_hours );

% Plot limits
date_start = datenum(2004,12,1);
date_end = datenum(2005,2,1);

% POST PROCESS: Replace out-of-bounds moments of pred with WW3 
tic
for fh = 1:num_hours
    pred(fh).a1F = pred(fh).a1;
    pred(fh).b1F = pred(fh).b1;
    pred(fh).a2F = pred(fh).a2;
    pred(fh).b2F = pred(fh).b2;
    for ff = 1:length(pred(fh).fr)
        inds = abs(pred(fh).a1(:,ff)./pred(fh).e(:,ff)) > 1;        
        pred(fh).a1F(inds,ff) = ww3.a1(inds,ff);
        
        inds = abs(pred(fh).b1(:,ff)./pred(fh).e(:,ff)) > 1;
        pred(fh).b1F(inds,ff) = ww3.b1(inds,ff);
        
        inds = abs(pred(fh).a2(:,ff)./pred(fh).e(:,ff)) > 1;      
        pred(fh).a2F(inds,ff) = ww3.a2(inds,ff);
        
        inds = abs(pred(fh).b2(:,ff)./pred(fh).e(:,ff)) > 1;       
        pred(fh).b2F(inds,ff) = ww3.b2(inds,ff);
    end
end
toc

% Estimate directional integrated for a freq band
abnd = 1:13; %0.04-0.105 Hz
[obs.md1SW,obs.md2SW,obs.spr1SW,obs.spr2SW,~,~]=getkuikstats(sum(obs.a1(:,abnd),2)./sum(obs.e(:,abnd),2),sum(obs.b1(:,abnd),2)./sum(obs.e(:,abnd),2),sum(obs.a2(:,abnd),2)./sum(obs.e(:,abnd),2),sum(obs.b2(:,abnd),2)./sum(obs.e(:,abnd),2));
[ww3.md1SW,ww3.md2SW,ww3.spr1SW,ww3.spr2SW,~,~]=getkuikstats(sum(ww3.a1(:,abnd),2)./sum(ww3.e(:,abnd),2),sum(ww3.b1(:,abnd),2)./sum(ww3.e(:,abnd),2),sum(ww3.a2(:,abnd),2)./sum(ww3.e(:,abnd),2),sum(ww3.b2(:,abnd),2)./sum(ww3.e(:,abnd),2));
for hr = 1:num_hours
    [pred(hr).md1SW,pred(hr).md2SW,pred(hr).spr1SW,pred(hr).spr2SW,~,~]=getkuikstats(sum(pred(hr).a1(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).b1(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).a2(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).b2(:,abnd),2)./sum(pred(hr).e(:,abnd),2));
    [pred(hr).md1SWF,pred(hr).md2SWF,pred(hr).spr1SWF,pred(hr).spr2SWF,~,~]=getkuikstats(sum(pred(hr).a1F(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).b1F(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).a2F(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).b2F(:,abnd),2)./sum(pred(hr).e(:,abnd),2));
end
abnd = 14:28; %0.105 - 0.25 Hz
[obs.md1SS,obs.md2SS,obs.spr1SS,obs.spr2SS,~,~]=getkuikstats(sum(obs.a1(:,abnd),2)./sum(obs.e(:,abnd),2),sum(obs.b1(:,abnd),2)./sum(obs.e(:,abnd),2),sum(obs.a2(:,abnd),2)./sum(obs.e(:,abnd),2),sum(obs.b2(:,abnd),2)./sum(obs.e(:,abnd),2));
[ww3.md1SS,ww3.md2SS,ww3.spr1SS,ww3.spr2SS,~,~]=getkuikstats(sum(ww3.a1(:,abnd),2)./sum(ww3.e(:,abnd),2),sum(ww3.b1(:,abnd),2)./sum(ww3.e(:,abnd),2),sum(ww3.a2(:,abnd),2)./sum(ww3.e(:,abnd),2),sum(ww3.b2(:,abnd),2)./sum(ww3.e(:,abnd),2));
for hr = 1:num_hours
    [pred(hr).md1SS,pred(hr).md2SS,pred(hr).spr1SS,pred(hr).spr2SS,~,~]=getkuikstats(sum(pred(hr).a1(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).b1(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).a2(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).b2(:,abnd),2)./sum(pred(hr).e(:,abnd),2));
    [pred(hr).md1SSF,pred(hr).md2SSF,pred(hr).spr1SSF,pred(hr).spr2SSF,~,~]=getkuikstats(sum(pred(hr).a1F(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).b1F(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).a2F(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).b2F(:,abnd),2)./sum(pred(hr).e(:,abnd),2));
end


%% Estimate RMSE:
diffcut = 90;
temp = ww3.md1SW-obs.md1SW;
temp(temp>diffcut) = NaN;
rmse_ww3_SW = sqrt(nanmean(temp.^2));
for fh = 1:num_hours
    temp = pred(fh).md1SWF-obs.md1SW;
    temp(temp>diffcut) = NaN;
    rmse_pred_SW(fh) = sqrt(nanmean(temp.^2));
end
temp = ww3.md1SS-obs.md1SS;
temp(temp>diffcut) = NaN;
rmse_ww3_SS = sqrt(nanmean(temp.^2));
for fh = 1:num_hours
    temp = pred(fh).md1SSF-obs.md1SS;
    temp(temp>diffcut) = NaN;
    rmse_pred_SS(fh) = sqrt(nanmean(temp.^2));
end

mycolors = lines(5);

clf
subplot(121)
hold on
lh(1) = plot(1:num_hours,rmse_pred_SW,'Color',mycolors(1,:));
plot(1:num_hours,rmse_ww3_SW*ones(1,num_hours),'--','Color',mycolors(1,:))
lh(2) = plot(1:num_hours,rmse_pred_SS,'Color',mycolors(2,:));
plot(1:num_hours,rmse_ww3_SS*ones(1,num_hours),'--','Color',mycolors(2,:))
legend(lh,'Swell','Seas')
ylabel('\theta_m RMSE [^o]')
xlabel('Forecast hour')
grid on
box on

% Next plot show a metric of fitting spectral shape
% Calc %different of spectra shape at each time step and average
abnd = 1:13;
bbnd = 14:28;
spdf_ww3_sw = nanmean(sum(abs(ww3.e(:,abnd)-obs.e(:,abnd)),2)./sum(obs.e(:,abnd),2));
spdf_ww3_ss = nanmean(sum(abs(ww3.e(:,bbnd)-obs.e(:,bbnd)),2)./sum(obs.e(:,bbnd),2));
spdf_pred_sw = NaN(1,num_hours);
spdf_pred_ss = NaN(1,num_hours);
for fh = 1:num_hours
spdf_pred_sw(fh) = nanmean(sum(abs(pred(fh).e(:,abnd)-obs.e(:,abnd)),2)./sum(obs.e(:,abnd),2));
spdf_pred_ss(fh) = nanmean(sum(abs(pred(fh).e(:,bbnd)-obs.e(:,bbnd)),2)./sum(obs.e(:,bbnd),2));
end

subplot(122)
hold on
lh(1) = plot(1:num_hours,spdf_pred_sw,'Color',mycolors(1,:));
plot(1:num_hours,spdf_ww3_sw*ones(1,num_hours),'--','Color',mycolors(1,:))
lh(2) = plot(1:num_hours,spdf_pred_ss,'Color',mycolors(2,:));
plot(1:num_hours,spdf_ww3_ss*ones(1,num_hours),'--','Color',mycolors(2,:))
legend(lh,'Swell','Seas')
ylabel('Spectral Misfit [%]')
xlabel('Forecast hour')
grid on
box on


printFig(gcf,strcat(buoy,'fig_dir_rmse'),[8 4],'png')



return
%%
tt = 3900;
fh = 3;

% clf
% plot(obs.fr,obs.e(tt,:))
% hold on
% plot(ww3.fr,ww3.e(tt,:))
% plot(pred(fh).fr,pred(fh).e(tt,:))










