clearvars
M = load('cfsr_buoy_met_pred.mat');
O = load('../met_buoy_obs/buoy_bulkwave_met_obs_qc.mat');

%%
clf

for ii = 1:21

x = M.wndspd(:,ii);
y = interp1(O.time,O.wndspd(ii,:),M.time);

mylim = 25;
subplot(3,7,ii)
I = ~isnan(x) & ~isnan(y);
dscatter(y(I),x(I))
grid on
hold on
plot([0 mylim],[0 mylim],'-k')
xlabel('Obs wind speed [m/s]')
ylabel('Pred wind speeds [m/s]')
title(O.id{ii})
end

packfig(3,7)
printFig(gcf,'scatter_wind_speed',[18 7],'png')

%%
clf
for ii = 1:21

x = wrapTo360(M.wnddir(:,ii));
y = interp1(O.time,O.wnddir(ii,:),M.time);

mylim = 360;
subplot(3,7,ii)
%plot(x,y,'.')
I = ~isnan(x) & ~isnan(y);
dscatter(y(I),x(I))
grid on
hold on
plot([0 mylim],[0 mylim],'-k')
xlabel('Obs wind speed [m/s]')
ylabel('Pred wind speeds [m/s]')
text(120,310,O.id{ii},'Color','w')
xlim([0 mylim])
ylim([0 mylim])
end

packfig(3,7)
printFig(gcf,'scatter_wind_dir',[18 7],'png')











