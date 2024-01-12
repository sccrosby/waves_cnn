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
myylimtm = [0 0.8];

pleft = .1;
pright = .05;
ptop = .05;
pbot = .1;
pspace = .01;
pwid = (1-pleft-pright-2*pspace)/3;
pheight = (1-ptop-pbot-pspace)/3;

PLT = PLOTTER;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot HS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%
% Hs column 1 %
%%%%%%%%%%%%%%%
axes('position',[pleft+0*(pwid+pspace) pbot+2*(pheight+pspace) pwid pheight])
title('Grays Harbor')
hold on


PLT.plot_inter_buoy_hs_seas(PLT, buoy_46211, buoy_46211__46214, buoy_46211__46218, mycolors)

ylim(myylimhs)
ylabel('H_s RMSE [m]')


% Legend falseplots
hold on
clear ph
ph(1)=plot([0 5],[-5 -5],'-','Color',PLT.ww3_color,'MarkerFaceColor',PLT.ww3_color,'LineWidth',1.0);
ph(2)=plot([0 5],[-5 -5],PLT.ww3_d_line,'Color',PLT.ww3_adj_color,'MarkerFaceColor',PLT.ww3_adj_color,'LineWidth',1.0);
ph(3)=plot([0 5],[-5 -5],'-','Color',mycolors(1,:),'MarkerFaceColor',mycolors(1,:),'LineWidth',1.0);
ph(4)=plot([0 5],[-5 -5],'-','Color',mycolors(2,:),'MarkerFaceColor',mycolors(2,:),'LineWidth',1.0);
ph(5)=plot([0 5],[-5 -5],'-','Color',mycolors(3,:),'MarkerFaceColor',mycolors(3,:),'LineWidth',1.0);

lh = legend(ph, 'WW3', 'WW3-debiased', 'Grays Harbor SWRL Net', 'Point Reyes SWRL Net', 'Harvest SWRL Net');
set(lh,'box','off')
% set(lh,'Location','southeast')
set(lh,'Position',[0.276 0.74 0 0])

text(0.5, 0.01, '(a)', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'fontsize', 8);



%%%%%%%%%%%%%%%
% Hs column 2 %
%%%%%%%%%%%%%%%
axes('position',[pleft+1*(pwid+pspace) pbot+2*(pheight+pspace) pwid pheight])
title('Point Reyes')
hold on
PLT.plot_inter_buoy_hs_seas(PLT, buoy_46214__46211, buoy_46214, buoy_46214__46218, mycolors)
ylim(myylimhs)

set(gca,'YTickLabel',[]);

text(0.5, 0.01, '(b)', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'fontsize', 8);


%%%%%%%%%%%%%%%
% Hs column 3 %
%%%%%%%%%%%%%%%
axes('position',[pleft+2*(pwid+pspace) pbot+2*(pheight+pspace) pwid pheight])
title('Harvest')
hold on
PLT.plot_inter_buoy_hs_seas(PLT, buoy_46218__46211, buoy_46218__46214, buoy_46218, mycolors)
ylim(myylimhs)

set(gca,'YTickLabel',[]);

text(0.5, 0.01, '(c)', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'fontsize', 8);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Theta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%
% Th column 1 %
%%%%%%%%%%%%%%%
axes('position',[pleft+0*(pwid+pspace) pbot+1*(pheight+pspace) pwid pheight])
hold on

PLT.plot_inter_buoy_theta_seas(PLT, buoy_46211, buoy_46211__46214, buoy_46211__46218, mycolors, num_hours)

% other figure settings
ylabel('\theta_m RMSE [^o]')
xlabel('Forecast hour')
ylim(myylim)
set(gca,'YTick',0:5:10)

text(0.5, 0.2, '(d)', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'fontsize', 8);


%%%%%%%%%%%%%%%
% Th column 2 %
%%%%%%%%%%%%%%%
axes('position',[pleft+1*(pwid+pspace) pbot+1*(pheight+pspace) pwid pheight])
hold on

PLT.plot_inter_buoy_theta_seas(PLT, buoy_46214__46211, buoy_46214, buoy_46214__46218, mycolors, num_hours)
ylim(myylim)
set(gca,'YTickLabel',[]);

text(0.5, 0.2, '(e)', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'fontsize', 8);


%%%%%%%%%%%%%%%
% Th column 3 %
%%%%%%%%%%%%%%%
axes('position',[pleft+2*(pwid+pspace) pbot+1*(pheight+pspace) pwid pheight])
hold on

PLT.plot_inter_buoy_theta_seas(PLT, buoy_46218__46211, buoy_46218__46214, buoy_46218, mycolors, num_hours)
ylim(myylim)

set(gca,'YTickLabel',[]);

text(0.5, 0.2, '(f)', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'fontsize', 8);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Mean Period
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%
% Tm column 1 %
%%%%%%%%%%%%%%%
axes('position',[pleft+0*(pwid+pspace) pbot+0*(pheight+pspace) pwid pheight])
hold on

PLT.plot_inter_buoy_period_seas(PLT, buoy_46211, buoy_46211__46214, buoy_46211__46218, mycolors)

ylim(myylimtm)


ylabel('T_m RMSE [sec]');

text(0.5, 0.025, '(g)', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'fontsize', 8);




%%%%%%%%%%%%%%%
% Tm column 2 %
%%%%%%%%%%%%%%%
axes('position',[pleft+1*(pwid+pspace) pbot+0*(pheight+pspace) pwid pheight])
hold on

PLT.plot_inter_buoy_period_seas(PLT, buoy_46214__46211, buoy_46214, buoy_46214__46218, mycolors)

ylim(myylimtm)

set(gca,'YTickLabel',[]);

text(0.5, 0.025, '(h)', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'fontsize', 8);


%%%%%%%%%%%%%%%
% Tm column 3 %
%%%%%%%%%%%%%%%
axes('position',[pleft+2*(pwid+pspace) pbot+0*(pheight+pspace) pwid pheight])
hold on

PLT.plot_inter_buoy_period_seas(PLT, buoy_46218__46211, buoy_46218__46214, buoy_46218, mycolors)
ylim(myylimtm)

set(gca,'YTickLabel',[]);

text(0.5, 0.025, '(i)', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'fontsize', 8);


%%%
% FINALLY
%%%%
printFig(gcf,'fig_07_inter_buoy_seas',[6.5 6],'pdf')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot HS function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function hs_p = plot_hs_swell(buoy_6, buoy_12, buoy_24, buoy_48)
%     plot(buoy_6.hs_sw_rmse_pred,swell_line,'Color',mycolors(1,:))
%     plot(buoy_12.hs_sw_rmse_pred,swell_line,'Color',mycolors(2,:))
%     plot(buoy_24.hs_sw_rmse_pred,swell_line,'Color',mycolors(3,:))
%     plot(buoy_48.hs_sw_rmse_pred,swell_line,'Color',mycolors(4,:))
% end