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
pleft = .05;
pright = .03;
ptop = .12;
pbot = .1;
pspace = .01;
pwid = (1-pleft-pright-2*pspace)/3;
pheight = (1-ptop-pbot-1*pspace)/2;


%%%%%%%%%%%%%%%%%%%%%%% Load data AND PLOT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files=dir(fol1);
fname = files(4).name(1:17);
[ obs, ww3, pred] = load_data( fol1, fname, num_hours );

% Convert each frequency band to equivalent "wave height"
ww3.em = sqrt(ww3.e);%.*repmat(ww3.bw,[length(ww3.time) 1]));
obs.em = sqrt(obs.e);%.*repmat(obs.bw,[length(obs.time) 1]));
for hr = 1:num_hours
    pred(hr).em = sqrt(pred(hr).e);%.*repmat(pred(hr).bw,[length(pred(hr).time) 1]));
end

% Calc RMSE
for ff = 1:length(obs.fr)
    for hr = 1:num_hours
        rmse_pred(hr,ff) = sqrt(nanmean((pred(hr).e(:,ff)-obs.e(:,ff)).^2));
        rmse_pred_br(hr,ff) = sqrt(nanmean((pred(hr).e(:,ff)-obs.e(:,ff)-nanmean(pred(hr).e(:,ff)-obs.e(:,ff))).^2));
    end
    mean_ww3(ff) = nanmean(ww3.e(:,ff));
    mean_obs(ff) = nanmean(obs.e(:,ff));
    rmse_ww3(ff) = sqrt(nanmean((ww3.e(:,ff)-obs.e(:,ff)).^2));         
    rmse_ww3_br(ff) = sqrt(nanmean((ww3.e(:,ff)-obs.e(:,ff)-nanmean(ww3.e(:,ff)-obs.e(:,ff))).^2));
    
    meanb_ww3(ff) = nanmean(ww3.em(:,ff));
    meanb_obs(ff) = nanmean(obs.em(:,ff));
    rmseb_ww3(ff) = sqrt(nanmean((ww3.em(:,ff)-obs.em(:,ff)).^2));         
    rmseb_ww3_br(ff) = sqrt(nanmean((ww3.em(:,ff)-obs.em(:,ff)-nanmean(ww3.em(:,ff)-obs.em(:,ff))).^2));
       
    me_ww3(ff) = sqrt(nanmean(abs(ww3.e(:,ff)-obs.e(:,ff))));
    me_ww3_br(ff) = sqrt(nanmean(abs(ww3.e(:,ff)-obs.e(:,ff)-nanmean(ww3.e(:,ff)-obs.e(:,ff)))));
    m_ww3(ff) = sqrt(nanmean(abs(ww3.e(:,ff).^2)));
end

% Estimate relative RMSE loss/gain
for ff = 1:length(obs.fr)
    for hr = 1:num_hours
        rmse_loss(hr,ff) = (rmse_pred(hr,ff)-rmse_ww3(ff))/rmse_ww3(ff);
        rmse_loss_br(hr,ff) = (rmse_pred(hr,ff)-rmse_ww3_br(ff))/rmse_ww3_br(ff);
    end
end

% SWRL NET correction
axes('position',[pleft+0*(pwid+pspace) pbot+1*(pheight+pspace) pwid pheight])
ax = contourf(obs.fr,1:num_hours,100*rmse_loss_br,20,'EdgeColor','none');
shading flat
colormap(mycolors)
caxis([-50 50])
grid on
ylabel('Forecast time [Hr]')
%xlabel('Frequency [Hz]')
set(gca,'XTickLabel',[])

ss_sw_divider = line([0.105 0.105], get(gca, 'ylim'), 'Color', 'black', 'LineStyle', '--');
t = text(0.245, 23.5, '(a) Grays Harbor', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'fontsize', 8);

text(0.12, 13, 'seas');
text(0.05, 13, 'swell');

% WW3 Absolute error
ax2 = axes('position',[pleft+0*(pwid+pspace) pbot+0*(pheight+pspace) pwid pheight]);
% plot(obs.fr,mean_obs,'k');
% hold on
% plot(obs.fr,mean_ww3);
% plot(obs.fr,rmse_ww3_br);
plot(obs.fr,mean_obs,'k');
hold on
plot(obs.fr,mean_ww3);
plot(obs.fr,rmse_ww3);
shading flat
colormap(mycolors)
%caxis([-50 50])
grid on
ylabel('Energy [m^2/Hz]')
xlabel('Frequency [Hz]')
lh=legend('Observed Mean Energy','WW3 Mean Energy','WW3 RMS Energy','WW3 RMS Energy Error');
set(lh,'box','off')
set(lh,'Position',[0.23 0.44 0 0])
xlim([.04 .25])
temp = get(ax2,'YTick');
set(ax2,'YTick',temp(1:end-1))

ss_sw_divider = line([0.105 0.105], get(gca, 'ylim'), 'Color', 'black', 'LineStyle', '--');
set(gca,'XTick',0.05:.05:.2);


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
    me_ww3(ff) = sqrt(nanmean(abs(ww3.e(:,ff)-obs.e(:,ff))));
    me_ww3_br(ff) = sqrt(nanmean(abs(ww3.e(:,ff)-obs.e(:,ff)-nanmean(ww3.e(:,ff)-obs.e(:,ff)))));    
end

% Estimate relative RMSE loss/gain
for ff = 1:length(obs.fr)
    for hr = 1:num_hours
        rmse_loss(hr,ff) = (rmse_pred(hr,ff)-rmse_ww3(ff))/rmse_ww3(ff);
        rmse_loss_br(hr,ff) = (rmse_pred(hr,ff)-rmse_ww3_br(ff))/rmse_ww3_br(ff);
    end
end

% SWRL NET correction
axes('position',[pleft+1*(pwid+pspace) pbot+1*(pheight+pspace) pwid pheight])
ax = contourf(obs.fr,1:num_hours,100*rmse_loss_br,20,'EdgeColor','none');
shading flat
colormap(mycolors)
caxis([-50 50])
grid on
%ylabel('Forecast time [Hr]')
%xlabel('Frequency [Hz]')
set(gca,'XTickLabel',[])
set(gca,'YTickLabel',[])

ss_sw_divider = line([0.105 0.105], get(gca, 'ylim'), 'Color', 'black', 'LineStyle', '--');
t = text(0.245, 23.5, '(b) Point Reyes', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'fontsize', 8);

% WW3 Absolute error
ax2 = axes('position',[pleft+1*(pwid+pspace) pbot+0*(pheight+pspace) pwid pheight]);
% plot(obs.fr,mean_obs,'k');
% hold on
% plot(obs.fr,mean_ww3);
% plot(obs.fr,rmse_ww3_br);
plot(obs.fr,mean_obs,'k');
hold on
plot(obs.fr,mean_ww3);
plot(obs.fr,rmse_ww3);
shading flat
colormap(mycolors)
%caxis([-50 50])
grid on
%ylabel('Energy [m^2/Hz]')
xlabel('Frequency [Hz]')
%lh=legend('WW3 Mean','WW3 RMSE');
%set(lh,'box','off')
xlim([.04 .25])
temp = get(ax2,'YTick');
set(ax2,'YTick',temp(1:end-1))
set(gca,'YTickLabel',[])

ss_sw_divider = line([0.105 0.105], get(gca, 'ylim'), 'Color', 'black', 'LineStyle', '--');
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
    me_ww3(ff) = sqrt(nanmean(abs(ww3.e(:,ff)-obs.e(:,ff))));
    me_ww3_br(ff) = sqrt(nanmean(abs(ww3.e(:,ff)-obs.e(:,ff)-nanmean(ww3.e(:,ff)-obs.e(:,ff)))));    
end

% Estimate relative RMSE loss/gain
for ff = 1:length(obs.fr)
    for hr = 1:num_hours
        rmse_loss(hr,ff) = (rmse_pred(hr,ff)-rmse_ww3(ff))/rmse_ww3(ff);
        rmse_loss_br(hr,ff) = (rmse_pred(hr,ff)-rmse_ww3_br(ff))/rmse_ww3_br(ff);
    end
end

% SWRL NET correction
axes('position',[pleft+2*(pwid+pspace) pbot+1*(pheight+pspace) pwid pheight])
ax = contourf(obs.fr,1:num_hours,100*rmse_loss_br,20,'EdgeColor','none');
shading flat
colormap(mycolors)
caxis([-50 50])
grid on
%ylabel('Forecast time [Hr]')
%xlabel('Frequency [Hz]')
set(gca,'XTickLabel',[])
set(gca,'YTickLabel',[])



ss_sw_divider = line([0.105 0.105], get(gca, 'ylim'), 'Color', 'black', 'LineStyle', '--');
t = text(0.245, 23.5, '(c) Harvest', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'fontsize', 8);


% WW3 Absolute error
ax2 = axes('position',[pleft+2*(pwid+pspace) pbot+0*(pheight+pspace) pwid pheight]);
% plot(obs.fr,mean_obs,'k');
% hold on
% plot(obs.fr,mean_ww3);
% plot(obs.fr,rmse_ww3_br);
plot(obs.fr,mean_obs,'k');
hold on
plot(obs.fr,mean_ww3);
plot(obs.fr,rmse_ww3);
shading flat
colormap(mycolors)
%caxis([-50 50])
grid on
%ylabel('Energy [m^2/Hz]')
xlabel('Frequency [Hz]')
%lh=legend('WW3 Mean','WW3 RMSE');
%set(lh,'box','off')
xlim([.04 .25])
temp = get(ax2,'YTick');
set(ax2,'YTick',temp(1:end-1))
set(gca,'YTickLabel',[])

ss_sw_divider = line([0.105 0.105], get(gca, 'ylim'), 'Color', 'black', 'LineStyle', '--');
set(gca,'XTick',0.05:.05:.2);
    
chan = colorbar('Location','NorthOutside','position',[pleft pbot+2*pheight+pspace+.01 1-pleft-pright .02]);
ylabel(chan,'Change in RMSE [%]')




% figures
printFig(gcf,'Figure04-rmse-fh',[6.5 5],'png')
% printFig(gcf,'fig_04_rmse_fh',[6.5 4],'png', 500)

%% POSTER PRINTING
% printFig(gcf,'poster_04_rmse_fh',[6.5 4],'png', 1000)


return

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Below are just validation checks, ignore, but don't delete %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Time Series View
clf
count = 0;
for ff = 1:2:length(obs.fr)-4
    count = count+1;
    ax(count)=subplot(12,1,count);
    plot(obs.time,obs.e(:,ff))
    hold on
    plot(ww3.time,ww3.e(:,ff))
    plot(pred(3).time,pred(3).e(:,ff))
    %plot(pred(6).time,pred(6).e(:,ff))
    %plot(pred(12).time,pred(12).e(:,ff))
end
legend('Obs','WW3','Pred-3','Pred-6','Pred-12')
linkaxes(ax,'x')
packfig(12,1)

%%
clf

count = 0;
for ff = [1 3 10 20 28]
    count = count+1;
subplot(2,3,count)
plot(obs.e(:,ff),ww3.e(:,ff),'.')
mym = nanmean(obs.e(:,ff));
rmse=sqrt(nanmean((obs.e(:,ff)-ww3.e(:,ff)).^2));
hold on
plot(xlim,[rmse rmse],'-k')
plot(xlim,[mym mym],'--k')
title(sprintf('fr = %4.3f Hz, rmse = %4.2f m^2/Hz',obs.fr(ff),rmse))
ylabel('Predicted E [m^2/Hz]')
xlabel('Observed E [m^2/Hz]')
end

subplot(2,3,count+1)
hso = 4*sqrt(obs.e*obs.bw');
hsp = 4*sqrt(ww3.e*ww3.bw');
plot(hso,hsp,'.')
rmse=sqrt(nanmean((hso-hsp).^2));
hold on
mym = nanmean(hso);
plot(xlim,[rmse rmse],'-k')
plot(xlim,[mym mym],'--k')
title(sprintf('rmse = %4.2f m',rmse))
ylabel('Predicted Hs [m]')
xlabel('Observed Hs [m]')


printFig(gcf,'NOAA42618_ScatterEnergy_Linear',[10 6],'png')


%%
clf

count = 0;
for ff = [1 3 10 20 28]
    count = count+1;
subplot(2,3,count)
loglog(obs.e(:,ff),ww3.e(:,ff),'.')
mym = nanmean(obs.e(:,ff));
rmse=sqrt(nanmean((obs.e(:,ff)-ww3.e(:,ff)).^2));
hold on
plot(xlim,[rmse rmse],'-k')
plot(xlim,[mym mym],'--k')
title(sprintf('fr = %4.3f Hz, rmse = %4.2f m^2/Hz',obs.fr(ff),rmse))
ylabel('Predicted E [m^2/Hz]')
xlabel('Observed E [m^2/Hz]')
end

subplot(2,3,count+1)
hso = 4*sqrt(obs.e*obs.bw');
hsp = 4*sqrt(ww3.e*ww3.bw');
loglog(hso,hsp,'.')
rmse=sqrt(nanmean((hso-hsp).^2));
hold on
mym = nanmean(hso);
plot(xlim,[rmse rmse],'-k')
plot(xlim,[mym mym],'--k')
title(sprintf('rmse = %4.2f m',rmse))
ylabel('Predicted Hs [m]')
xlabel('Observed Hs [m]')


printFig(gcf,'NOAA42618_ScatterEnergy_Log',[10 6],'png')

%%
clf

count = 0;
for ff = [1 3 10 20 28]
    count = count+1;
subplot(2,3,count)
loglog(obs.e(:,ff),ww3.em(:,ff),'.')
mym = nanmean(obs.em(:,ff));
rmse=sqrt(nanmean((obs.em(:,ff)-ww3.em(:,ff)).^2));
hold on
plot(xlim,[rmse rmse],'-k')
plot(xlim,[mym mym],'--k')
title(sprintf('fr = %4.3f Hz, rmse = %4.2f m/Hz',obs.fr(ff),rmse))
ylabel('Predicted E [m/Hz]')
xlabel('Observed E [m/Hz]')
end

subplot(2,3,count+1)
hso = 4*sqrt(obs.e*obs.bw');
hsp = 4*sqrt(ww3.e*ww3.bw');
loglog(hso,hsp,'.')
rmse=sqrt(nanmean((hso-hsp).^2));
hold on
mym = nanmean(hso);
plot(xlim,[rmse rmse],'-k')
plot(xlim,[mym mym],'--k')
title(sprintf('rmse = %4.2f m',rmse))
ylabel('Predicted Hs [m]')
xlabel('Observed Hs [m]')


printFig(gcf,'NOAA42618_ScatterEnergy_Log_m',[10 6],'png')
