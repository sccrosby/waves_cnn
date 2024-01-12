% Figure 
clearvars

% Global set
set(0,'defaultaxesfontsize',8)
TITLE = 'Inter Hour Comparison (Swells)';

% set data directory
data_dir = set_data_dir();

% other variables
num_hours = 24;
inp_hours = ['6', '12', '24', '48'];
rmse_vars = ['hs_sw_rmse_ww3', 'hs_sw_rmse_ww3_br', 'hs_sw_rmse_pred', ...
    'hs_ss_rmse_ww3', 'hs_ss_rmse_ww3_br', 'hs_ss_rmse_pred', ...
    'rmse_pred_SW', 'rmse_ww3_SW', 'rmse_pred_SS', 'rmse_ww3_SS'];


% set buoy structs for plotting
buoy_46211_6 = get_rmse_struct(strcat(data_dir, '46211_20190811014904_dev_test_6-24_2019_08_22_091752'));
buoy_46211_12 = get_rmse_struct(strcat(data_dir, '46211_20190728185615_dev_test_12-24_2019_08_22_091917'));
buoy_46211_24 = get_rmse_struct(strcat(data_dir, '46211_20190807190103_dev_test_24-24_2019_08_22_095340'));
buoy_46211_48 = get_rmse_struct(strcat(data_dir, '46211_20190804232133_dev_test_48-24_2019_08_22_092309'));

buoy_46214_6 = get_rmse_struct(strcat(data_dir, '46214_20190809181416_dev_test_6-24_2019_08_22_085425'));
buoy_46214_12 = get_rmse_struct(strcat(data_dir, '46214_20190806130859_dev_test_12-24_2019_08_22_085627'));
buoy_46214_24 = get_rmse_struct(strcat(data_dir, '46214_20190810195637_dev_test_24-24_2019_08_22_090350'));
buoy_46214_48 = get_rmse_struct(strcat(data_dir, '46214_20190802173750_dev_test_48-24_2019_08_22_091523'));

buoy_46218_6 = get_rmse_struct(strcat(data_dir, '46218_20190808160557_dev_test_6-24_2019_08_22_084556'));
buoy_46218_12 = get_rmse_struct(strcat(data_dir, '46218_20190813172439_dev_test_12-24_2019_08_22_084754'));
buoy_46218_24 = get_rmse_struct(strcat(data_dir, '46218_20190810121138_dev_test_24-24_2019_08_22_084934'));
buoy_46218_48 = get_rmse_struct(strcat(data_dir, '46218_20190805055624_dev_test_48-24_2019_08_22_085234'));

%%
% set figure details
clf
mycolors = lines(5);
myylim = [0 15];
myylimhs = [0 .4];

pleft = .1;
pright = .05;
ptop = .05;
pbot = .1;
pspace = .01;
pwid = (1-pleft-pright-2*pspace)/3;
pheight = (1-ptop-pbot-pspace)/2;

PLT = PLOTTER;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot HS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%
% Hs column 1 %
%%%%%%%%%%%%%%%
axes('position',[pleft+0*(pwid+pspace) pbot+1*(pheight+pspace) pwid pheight])
title('46211 - Grays Harbor')
hold on

PLT.plot_inter_hr_hs_seas(PLT, buoy_46211_6, buoy_46211_12, buoy_46211_24, buoy_46211_48, mycolors)
ylim(myylimhs)
ylabel('H_s RMSE [m]')

% Legend falseplots
ph(1)=plot([0 10],[-5 -5],'s','Color',PLT.ww3_color,'MarkerFaceColor',PLT.ww3_color);
ph(2)=plot([0 10],[-5 -5],'s','Color',PLT.ww3_adj_color,'MarkerFaceColor',PLT.ww3_adj_color);
ph(3)=plot([0 10],[-5 -5],'s','Color',mycolors(1,:),'MarkerFaceColor',mycolors(1,:));
ph(4)=plot([0 10],[-5 -5],'s','Color',mycolors(2,:),'MarkerFaceColor',mycolors(2,:));
ph(5)=plot([0 10],[-5 -5],'s','Color',mycolors(3,:),'MarkerFaceColor',mycolors(3,:));
ph(6)=plot([0 10],[-5 -5],'s','Color',mycolors(4,:),'MarkerFaceColor',mycolors(4,:));

lh = legend(ph, 'WW3','WW3-debiased','6-hour', '12-hour', '24-hour', '48-hour');
set(lh,'box','off')
set(lh,'Location','southeast')



%%%%%%%%%%%%%%%
% Hs column 2 %
%%%%%%%%%%%%%%%
axes('position',[pleft+1*(pwid+pspace) pbot+1*(pheight+pspace) pwid pheight])
title('46214 - Point Reyes')
hold on
PLT.plot_inter_hr_hs_seas(PLT, buoy_46214_6, buoy_46214_12, buoy_46214_24, buoy_46214_48, mycolors)
ylim(myylimhs)

set(gca,'YTickLabel',[]);

%%%%%%%%%%%%%%%
% Hs column 3 %
%%%%%%%%%%%%%%%
axes('position',[pleft+2*(pwid+pspace) pbot+1*(pheight+pspace) pwid pheight])
title('46218 - Harvest')
hold on
PLT.plot_inter_hr_hs_seas(PLT, buoy_46218_6, buoy_46218_12, buoy_46218_24, buoy_46218_48, mycolors)
ylim(myylimhs)

set(gca,'YTickLabel',[]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Theta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%
% Th column 1 %
%%%%%%%%%%%%%%%
axes('position',[pleft+0*(pwid+pspace) pbot pwid pheight])
hold on

PLT.plot_inter_hr_theta_seas(PLT, buoy_46211_6, buoy_46211_12, buoy_46211_24, buoy_46211_48, mycolors, num_hours)

% other figure settings
ylabel('\theta_m RMSE [^o]')
xlabel('Forecast hour')
ylim(myylim)
set(gca,'YTick',0:5:10)

%%%%%%%%%%%%%%%
% Th column 2 %
%%%%%%%%%%%%%%%
axes('position',[pleft+1*(pwid+pspace) pbot pwid pheight])
hold on

PLT.plot_inter_hr_theta_seas(PLT, buoy_46214_6, buoy_46214_12, buoy_46214_24, buoy_46214_48, mycolors, num_hours)
ylim(myylim)
set(gca,'YTickLabel',[]);

%%%%%%%%%%%%%%%
% Th column 3 %
%%%%%%%%%%%%%%%
axes('position',[pleft+2*(pwid+pspace) pbot pwid pheight])
hold on

PLT.plot_inter_hr_theta_seas(PLT, buoy_46218_6, buoy_46218_12, buoy_46218_24, buoy_46218_48, mycolors, num_hours)
ylim(myylim)

set(gca,'YTickLabel',[]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Mean Period
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%
% Tm column 1 %
%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%
% Tm column 2 %
%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%
% Tm column 3 %
%%%%%%%%%%%%%%%


%%%
% FINALLY
%%%%
printFig(gcf,'fig_inter_hour_seas',[8 5],'png')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot HS function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function hs_p = plot_hs_swell(buoy_6, buoy_12, buoy_24, buoy_48)
%     plot(buoy_6.hs_sw_rmse_pred,swell_line,'Color',mycolors(1,:))
%     plot(buoy_12.hs_sw_rmse_pred,swell_line,'Color',mycolors(2,:))
%     plot(buoy_24.hs_sw_rmse_pred,swell_line,'Color',mycolors(3,:))
%     plot(buoy_48.hs_sw_rmse_pred,swell_line,'Color',mycolors(4,:))
% end