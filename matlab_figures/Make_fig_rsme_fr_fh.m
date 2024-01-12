% Figure rsme_fr_fh
clearvars
addpath cbrewer

% Global set
set(0,'defaultaxesfontsize',8)

% Pick 3 models to plot
% Generalization models for 46211
% fol1 = 'paper_46211_20190525103359_dev_predictions';
% fol2 = 'generalization_46214_46211_20190525103359_dev_test_24-24_2019_05_30_184624';
% fol3 = 'generalization_46218_46211_20190525103359_dev_test_24-24_2019_05_30_185138';


% % Pick 3 models to plot
% % Generalization models for 46214
% fol1 = 'generalization_46211_46214_20190525165851_dev_test_24-24_2019_05_30_185718';
% fol2 = 'paper_46214_20190525165851_test_predictions';
% fol3 = 'generalization_46218_46214_20190525165851_dev_test_24-24_2019_05_30_185946';
% 
% 
% % Pick 3 models to plot
% % Generalization models for 46218
% fol1 = 'generalization_46211_46218_20190525201735_dev_test_24-24_2019_05_30_190707';
% fol2 = 'generalization_46214_46218_20190525201735_dev_test_24-24_2019_05_30_191041';
% fol3 = 'paper_46218_20190525201735_test_predictions';


% CUSTOM CHOICES Pick 3 models to plot
fol1 = 'paper_46211_20190525103359_dev_predictions'
fol2 = '46211_20190601094513_dev_dev_24-24_2019_06_01_102532'
fol3 = '46211_20190601094513_dev_test_24-24_2019_06_01_102557'


% built-in for jonny to run this on a linux machine
if isunix && strfind(pwd, 'jonny')
    fol1 = strcat('/home/hutch_research/projects/ml_waves19_jonny/data_outputs/',fol1);
    fol2 = strcat('/home/hutch_research/projects/ml_waves19_jonny/data_outputs/',fol2);
    fol3 = strcat('/home/hutch_research/projects/ml_waves19_jonny/data_outputs/',fol3);
    addpath /home/hutch_research/projects/ml_waves19/MatlabPlottingCode/cbrewer
end


% Custom colormap using cbrewer (Red-Blue with white in middle)
mycolors = flipud(cbrewer('div','RdBu',101));

num_hours = 24;

clf
pleft = .07;
pright = .05;
ptop = .15;
pbot = .1;
pspace = .01;
pwid = (1-pleft-pright-2*pspace)/3;
pheight = 1-ptop-pbot;


%%%%%%%%%%%%%%%%%%%%%%% Load data AND PLOT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files=dir(fol1);
fname = files(4).name(1:17);
[ obs, ww3, pred] = load_data( fol1, fname, num_hours );

% Calc RMSE
for ff = 1:length(obs.fr)
    for hr = 1:num_hours
        rmse_pred(hr,ff) = sqrt(nanmean((pred(hr).e(:,ff)-obs.e(:,ff)).^2));
        rmse_pred_br(hr,ff) = sqrt(nanmean((pred(hr).e(:,ff)-obs.e(:,ff)-nanmean(pred(hr).e(:,ff)-obs.e(:,ff))).^2));
    end
    rmse_ww3(ff) = sqrt(nanmean((ww3.e(:,ff)-obs.e(:,ff)).^2));
    rmse_ww3_br(ff) = sqrt(nanmean((ww3.e(:,ff)-obs.e(:,ff)-nanmean(ww3.e(:,ff)-obs.e(:,ff))).^2));
end

% Estimate relative RMSE loss/gain
for ff = 1:length(obs.fr)
    for hr = 1:num_hours
        rmse_loss(hr,ff) = (rmse_pred(hr,ff)-rmse_ww3(ff))/rmse_ww3(ff);
        rmse_loss_br(hr,ff) = (rmse_pred(hr,ff)-rmse_ww3_br(ff))/rmse_ww3_br(ff);
    end
end

axes('position',[pleft+0*(pwid+pspace) pbot pwid pheight])
ax = contourf(obs.fr,1:num_hours,100*rmse_loss_br,20,'EdgeColor','none');
shading flat
colormap(mycolors)
caxis([-50 50])
grid on
ylabel('Forecast time [Hr]')
xlabel('Frequency [Hz]')


%%%%%%%%%%%%%%%%%%%%%%% Load data AND PLOT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files=dir(fol2);
fname = files(4).name(1:17);
[ obs, ww3, pred] = load_data( fol2, fname, num_hours );

% Calc RMSE
for ff = 1:length(obs.fr)
    for hr = 1:num_hours
        rmse_pred(hr,ff) = sqrt(nanmean((pred(hr).e(:,ff)-obs.e(:,ff)).^2));
        rmse_pred_br(hr,ff) = sqrt(nanmean((pred(hr).e(:,ff)-obs.e(:,ff)-nanmean(pred(hr).e(:,ff)-obs.e(:,ff))).^2));
    end
    rmse_ww3(ff) = sqrt(nanmean((ww3.e(:,ff)-obs.e(:,ff)).^2));
    rmse_ww3_br(ff) = sqrt(nanmean((ww3.e(:,ff)-obs.e(:,ff)-nanmean(ww3.e(:,ff)-obs.e(:,ff))).^2));
end

% Estimate relative RMSE loss/gain
for ff = 1:length(obs.fr)
    for hr = 1:num_hours
        rmse_loss(hr,ff) = (rmse_pred(hr,ff)-rmse_ww3(ff))/rmse_ww3(ff);
        rmse_loss_br(hr,ff) = (rmse_pred(hr,ff)-rmse_ww3_br(ff))/rmse_ww3_br(ff);
    end
end

axes('position',[pleft+1*(pwid+pspace) pbot pwid pheight])
ax = contourf(obs.fr,1:num_hours,100*rmse_loss_br,20,'EdgeColor','none');
shading flat
colormap(mycolors)
caxis([-50 50])
grid on
%ylabel('Forecast time [Hr]')
xlabel('Frequency [Hz]')
set(gca,'YTickLabel',[])

%%%%%%%%%%%%%%%%%%%%%%% Load data AND PLOT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files=dir(fol3);
fname = files(4).name(1:17);
[ obs, ww3, pred] = load_data( fol3, fname, num_hours );

% Calc RMSE
for ff = 1:length(obs.fr)
    for hr = 1:num_hours
        rmse_pred(hr,ff) = sqrt(nanmean((pred(hr).e(:,ff)-obs.e(:,ff)).^2));
        rmse_pred_br(hr,ff) = sqrt(nanmean((pred(hr).e(:,ff)-obs.e(:,ff)-nanmean(pred(hr).e(:,ff)-obs.e(:,ff))).^2));
    end
    rmse_ww3(ff) = sqrt(nanmean((ww3.e(:,ff)-obs.e(:,ff)).^2));
    rmse_ww3_br(ff) = sqrt(nanmean((ww3.e(:,ff)-obs.e(:,ff)-nanmean(ww3.e(:,ff)-obs.e(:,ff))).^2));
end

% Estimate relative RMSE loss/gain
for ff = 1:length(obs.fr)
    for hr = 1:num_hours
        rmse_loss(hr,ff) = (rmse_pred(hr,ff)-rmse_ww3(ff))/rmse_ww3(ff);
        rmse_loss_br(hr,ff) = (rmse_pred(hr,ff)-rmse_ww3_br(ff))/rmse_ww3_br(ff);
    end
end

axes('position',[pleft+2*(pwid+pspace) pbot pwid pheight])
ax = contourf(obs.fr,1:num_hours,100*rmse_loss_br,20,'EdgeColor','none');
shading flat
colormap(mycolors)
caxis([-50 50])
grid on
%ylabel('Forecast time [Hr]')
xlabel('Frequency [Hz]')
set(gca,'YTickLabel',[])

    
chan = colorbar('Location','NorthOutside','position',[pleft pbot+pheight+.01 1-pleft-pright .02]);
ylabel(chan,'Change in RMSE [%]')
printFig(gcf,'fig_rmse_fh',[8.5 4],'png')





