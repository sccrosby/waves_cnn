% Figure 
clearvars

%%% TODO: wait until inter-hour-swells is done

% Global set
set(0,'defaultaxesfontsize',8)

% set data directory
data_dir = set_data_dir();

% other variables
num_hours = 24;
inp_hours = ['6', '12', '24', '48'];
rmse_vars = ['hs_sw_rmse_ww3', 'hs_sw_rmse_ww3_br', 'hs_sw_rmse_pred', ...
    'hs_ss_rmse_ww3', 'hs_ss_rmse_ww3_br', 'hs_ss_rmse_pred', ...
    'rmse_pred_SW', 'rmse_ww3_SW', 'rmse_pred_SS', 'rmse_ww3_SS'];


% set buoy structs for plotting
buoy_46211 = get_rmse_struct(strcat(data_dir, '46211_20190811014904_dev_test_6-24_2019_08_22_091752'));
buoy_46211__46214 = get_rmse_struct(strcat(data_dir, '46211__46214_20190809181416_test_6_24'));
buoy_46211__46218 = get_rmse_struct(strcat(data_dir, '46211__46218_20190808160557_test_6-24'));

buoy_46214 = get_rmse_struct(strcat(data_dir, '46214_20190809181416_dev_test_6-24_2019_08_22_085425'));
buoy_46214__46211 = get_rmse_struct(strcat(data_dir, '46214__46211_20190811014904_test_6_24'));
buoy_46214__46218 = get_rmse_struct(strcat(data_dir, '46214__46218_20190808160557_test_6_24'));

buoy_46218 = get_rmse_struct(strcat(data_dir, '46218_20190808160557_dev_test_6-24_2019_08_22_084556'));
buoy_46218__46211 = get_rmse_struct(strcat(data_dir, '46218__46211_20190811014904_test_6_24'));
buoy_46218__46214 = get_rmse_struct(strcat(data_dir, '46218__46214_20190809181416_test_6_24'));

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


PLT.plot_inter_buoy_hs_seas(PLT, buoy_46211, buoy_46211__46214, buoy_46211__46218, mycolors)

ylim(myylimhs)
ylabel('H_s RMSE [m]')


% Legend falseplots
hold on
clear ph
ph(1)=plot([0 10],[-5 -5],'-','Color',PLT.ww3_color,'MarkerFaceColor',PLT.ww3_color);
ph(2)=plot([0 10],[-5 -5],'-','Color',PLT.ww3_adj_color,'MarkerFaceColor',PLT.ww3_adj_color);
ph(3)=plot([0 10],[-5 -5],'-','Color',mycolors(1,:),'MarkerFaceColor',mycolors(1,:));
ph(4)=plot([0 10],[-5 -5],'-','Color',mycolors(2,:),'MarkerFaceColor',mycolors(2,:));
ph(5)=plot([0 10],[-5 -5],'-','Color',mycolors(3,:),'MarkerFaceColor',mycolors(3,:));

lh = legend(ph, 'WW3', 'WW3-debiased', '46211 model', '46214 model', '46218 model');
set(lh,'box','off')
set(lh,'Location','southeast')



%%%%%%%%%%%%%%%
% Hs column 2 %
%%%%%%%%%%%%%%%
axes('position',[pleft+1*(pwid+pspace) pbot+1*(pheight+pspace) pwid pheight])
title('46214 - Point Reyes')
hold on
PLT.plot_inter_buoy_hs_seas(PLT, buoy_46214__46211, buoy_46214, buoy_46214__46218, mycolors)
ylim(myylimhs)

set(gca,'YTickLabel',[]);

%%%%%%%%%%%%%%%
% Hs column 3 %
%%%%%%%%%%%%%%%
axes('position',[pleft+2*(pwid+pspace) pbot+1*(pheight+pspace) pwid pheight])
title('46218 - Harvest')
hold on
PLT.plot_inter_buoy_hs_seas(PLT, buoy_46218__46211, buoy_46218__46214, buoy_46218, mycolors)
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

PLT.plot_inter_buoy_theta_seas(PLT, buoy_46211, buoy_46211__46214, buoy_46211__46218, mycolors, num_hours)

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

PLT.plot_inter_buoy_theta_seas(PLT, buoy_46214__46211, buoy_46214, buoy_46214__46218, mycolors, num_hours)
ylim(myylim)
set(gca,'YTickLabel',[]);

%%%%%%%%%%%%%%%
% Th column 3 %
%%%%%%%%%%%%%%%
axes('position',[pleft+2*(pwid+pspace) pbot pwid pheight])
hold on

PLT.plot_inter_buoy_theta_seas(PLT, buoy_46218__46211, buoy_46218__46214, buoy_46218, mycolors, num_hours)
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
printFig(gcf,'fig_inter_buoy_seas',[8 5],'png')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot HS function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function hs_p = plot_hs_swell(buoy_6, buoy_12, buoy_24, buoy_48)
%     plot(buoy_6.hs_sw_rmse_pred,swell_line,'Color',mycolors(1,:))
%     plot(buoy_12.hs_sw_rmse_pred,swell_line,'Color',mycolors(2,:))
%     plot(buoy_24.hs_sw_rmse_pred,swell_line,'Color',mycolors(3,:))
%     plot(buoy_48.hs_sw_rmse_pred,swell_line,'Color',mycolors(4,:))
% end