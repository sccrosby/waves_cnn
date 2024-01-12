% Figure rsme_fr_fh
clearvars
addpath cbrewer

% Global set
set(0,'defaultaxesfontsize',8)

% set data_dir
[data_dir] = set_data_dir();

% Pick 3 models to plot
% Folders
fol1 = strcat(data_dir, '46211_20190811014904_dev_test_6-24_2019_08_22_091752');
fol2 = strcat(data_dir, '46214_20190809181416_dev_test_6-24_2019_08_22_085425');
fol3 = strcat(data_dir, '46218_20190808160557_dev_test_6-24_2019_08_22_084556');


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


ss_sw_divider = line([0.105 0.105], get(gca, 'ylim'), 'Color', 'black', 'LineStyle', '--');

t = text(0.245, 23.5, '(a) Grays Harbor', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'fontsize', 8);
set(gca,'XTick',0.05:.05:.2);

text(0.12, 13, 'seas');
text(0.05, 13, 'swell');


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
% title('46211 - Grays Harbor')
ax = contourf(obs.fr,1:num_hours,100*rmse_loss_br,20,'EdgeColor','none');
shading flat
colormap(mycolors)
caxis([-50 50])
grid on
%ylabel('Forecast time [Hr]')
xlabel('Frequency [Hz]')
set(gca,'YTickLabel',[])

ss_sw_divider = line([0.105 0.105], get(gca, 'ylim'), 'Color', 'black', 'LineStyle', '--');
t = text(0.245, 23.5, '(b) Point Reyes', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'fontsize', 8);
% t = text(0.05, 23.5, '(b)', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
set(gca,'XTick',0.05:.05:.2);

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
title('46218 - Harvest')
ax = contourf(obs.fr,1:num_hours,100*rmse_loss_br,20,'EdgeColor','none');
shading flat
colormap(mycolors)
caxis([-50 50])
grid on
%ylabel('Forecast time [Hr]')
xlabel('Frequency [Hz]')
set(gca,'YTickLabel',[])

ss_sw_divider = line([0.105 0.105], get(gca, 'ylim'), 'Color', 'black', 'LineStyle', '--');
t = text(0.245, 23.5, '(c) Harvest', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'fontsize', 8);
set(gca,'XTick',0.05:.05:.2);
    
chan = colorbar('Location','NorthOutside','position',[pleft pbot+pheight+.01 1-pleft-pright .02]);
ylabel(chan,'Change in RMSE [%]')
printFig(gcf,'fig_04_rmse_fh',[6.5 4],'png', 500)







