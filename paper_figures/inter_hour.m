% Figure 
clearvars

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
buoy_46211_6 = get_rmse_struct(strcat(data_dir, '46211_20190811014904_dev_test_6-24_2019_08_22_091752'));
buoy_46211_12 = get_rmse_struct(strcat(data_dir, '46211_20190728185615_dev_test_12-24_2019_08_22_091917'));
buoy_46211_24 = get_rmse_struct(strcat(data_dir, '46211_20190807190103_dev_test_24-24_2019_08_22_095340'));
buoy_46211_48 = get_rmse_struct(strcat(data_dir, '46211_20190804232133_dev_test_48-24_2019_08_22_092309'));


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

ww3_color = [0. 0. 0.];
ww3_adj_color = [.5 .5 .5];
sea_line = '-';
swell_line = '--';


% Plot HS
axes('position',[pleft+0*(pwid+pspace) pbot+1*(pheight+pspace) pwid pheight])
hold on

% WW3 Swell
plot([1 24],[buoy_46211_6.hs_sw_rmse_ww3 buoy_46211_6.hs_sw_rmse_ww3],swell_line,'Color',ww3_color)
plot([1 24],[buoy_46211_6.hs_sw_rmse_ww3_br buoy_46211_6.hs_sw_rmse_ww3_br],swell_line,'Color',ww3_adj_color)

% WW3 SEA
plot([1 24],[buoy_46211_6.hs_ss_rmse_ww3 buoy_46211_6.hs_ss_rmse_ww3],sea_line,'Color', ww3_color)
plot([1 24],[buoy_46211_6.hs_ss_rmse_ww3_br buoy_46211_6.hs_ss_rmse_ww3_br],sea_line,'Color', ww3_adj_color)

% Swell
plot(buoy_46211_6.hs_sw_rmse_pred,swell_line,'Color',mycolors(1,:))
plot(buoy_46211_12.hs_sw_rmse_pred,swell_line,'Color',mycolors(2,:))
plot(buoy_46211_24.hs_sw_rmse_pred,swell_line,'Color',mycolors(3,:))
plot(buoy_46211_48.hs_sw_rmse_pred,swell_line,'Color',mycolors(4,:))


% Seas
plot(buoy_46211_6.hs_ss_rmse_pred,sea_line,'Color',mycolors(1,:))
plot(buoy_46211_12.hs_ss_rmse_pred,sea_line,'Color',mycolors(2,:))
plot(buoy_46211_24.hs_ss_rmse_pred,sea_line,'Color',mycolors(3,:))
plot(buoy_46211_48.hs_ss_rmse_pred,sea_line,'Color',mycolors(4,:))

% Legend falseplots
ph(1)=plot([0 20],[-5 -5],swell_line,'Color',[0. 0. 0.]);
ph(2)=plot([0 10],[-5 -5],sea_line,'Color',[0. 0. 0.]);
ph(3)=plot([0 10],[-5 -5],'s','Color',ww3_adj_color,'MarkerFaceColor',ww3_adj_color);
ph(4)=plot([0 10],[-5 -5],'s','Color',ww3_color,'MarkerFaceColor',ww3_color);
ph(5)=plot([0 10],[-5 -5],'s','Color',mycolors(1,:),'MarkerFaceColor',mycolors(1,:));
ph(6)=plot([0 10],[-5 -5],'s','Color',mycolors(2,:),'MarkerFaceColor',mycolors(2,:));
ph(7)=plot([0 10],[-5 -5],'s','Color',mycolors(3,:),'MarkerFaceColor',mycolors(3,:));
ph(8)=plot([0 10],[-5 -5],'s','Color',mycolors(4,:),'MarkerFaceColor',mycolors(4,:));
lh = legend(ph,'Swell','Sea','WW3-debiased','WW3','6-hour', '12-hour', '24-hour', '48-hour');
set(lh,'box','off')
set(gca,'XTickLabel',[])

grid on
box on
ylabel('H_s RMSE [m]')
xlabel('Forecast hour')
ylim(myylimhs)

% Plot Theta
axes('position',[pleft+0*(pwid+pspace) pbot pwid pheight])
hold on

% WW3 lines
plot(1:num_hours,buoy_46211_6.rmse_ww3_SW*ones(1,num_hours),swell_line,'Color',ww3_color)
plot(1:num_hours,buoy_46211_6.rmse_ww3_SS*ones(1,num_hours),sea_line,'Color',ww3_color)

% Swells
plot(1:num_hours,buoy_46211_6.rmse_pred_SW,swell_line,'Color',mycolors(1,:));
plot(1:num_hours,buoy_46211_12.rmse_pred_SW,swell_line,'Color',mycolors(2,:));
plot(1:num_hours,buoy_46211_24.rmse_pred_SW,swell_line,'Color',mycolors(3,:));
plot(1:num_hours,buoy_46211_48.rmse_pred_SW,swell_line,'Color',mycolors(4,:));

% seas
plot(1:num_hours,buoy_46211_6.rmse_pred_SS,sea_line,'Color',mycolors(1,:));
plot(1:num_hours,buoy_46211_12.rmse_pred_SS,sea_line,'Color',mycolors(2,:));
plot(1:num_hours,buoy_46211_24.rmse_pred_SS,sea_line,'Color',mycolors(3,:));
plot(1:num_hours,buoy_46211_48.rmse_pred_SS,sea_line,'Color',mycolors(4,:));

ylabel('\theta_m RMSE [^o]')
xlabel('Forecast hour')
ylim(myylim)
set(gca,'YTick',0:5:10)
grid on
box on


