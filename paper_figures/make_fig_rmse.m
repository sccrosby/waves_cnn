% Figure 
clearvars

% Global set
set(0,'defaultaxesfontsize',8)

% data_dir
data_dir = set_data_dir();

% Pick 3 models to plot


% Dev
fol1 = strcat(data_dir, '46211_20190811014904_dev_test_6-24_2019_08_22_091752');
fol2 = strcat(data_dir, '46211_20190811014904_dev_test_6-24_2019_08_22_091752');
fol3 = strcat(data_dir, '46211_20190811014904_dev_test_6-24_2019_08_22_091752');

num_hours = 24;

clf
mycolors = lines(5);
myylim = [0 15];
myylimhs = [0 .4];
myylimtm = [0 1];

pleft = .1;
pright = .05;
ptop = .05;
pbot = .1;
pspace = .01;
pwid = (1-pleft-pright-2*pspace)/3;
pheight = (1-ptop-pbot-pspace)/3;

% First folder
[hs_sw_rmse_ww3, hs_sw_rmse_ww3_br, hs_sw_rmse_pred, hs_ss_rmse_ww3, ...
    hs_ss_rmse_ww3_br, hs_ss_rmse_pred, rmse_pred_SW, rmse_ww3_SW, ...
    rmse_pred_SS, rmse_ww3_SS, tm_sw_rmse_pred, tm_ss_rmse_pred, ...
    tm_sw_rmse_ww3, tm_ss_rmse_ww3] = get_rmse( fol1 );

% Plot HS
axes('position',[pleft+0*(pwid+pspace) pbot+2*(pheight+pspace) pwid pheight])
hold on
% Swell
plot([1 24],[hs_sw_rmse_ww3 hs_sw_rmse_ww3],'--','Color',mycolors(1,:))
plot([1 24],[hs_sw_rmse_ww3_br hs_sw_rmse_ww3_br],'-.','Color',mycolors(1,:))
plot(hs_sw_rmse_pred,'-','Color',mycolors(1,:))
% Seas
plot([1 24],[hs_ss_rmse_ww3 hs_ss_rmse_ww3],'--','Color',mycolors(2,:))
plot([1 24],[hs_ss_rmse_ww3_br hs_ss_rmse_ww3_br],'-.','Color',mycolors(2,:))
plot(hs_ss_rmse_pred,'-','Color',mycolors(2,:))

% Legend falseplots
ph(1)=plot([0 10],[-5 -5],'-','Color',[.7 .7 .7]);
ph(2)=plot([0 10],[-5 -5],'--','Color',[.7 .7 .7]);
ph(3)=plot([0 10],[-5 -5],'-.','Color',[.7 .7 .7]);
lh = legend(ph,'Adj','WW3','WW3-debiased','Location','SouthEast');
set(lh,'box','off')
set(gca,'XTickLabel',[])

grid on
box on
ylabel('H_s RMSE [m]')
xlabel('Forecast hour')
ylim(myylimhs)

% Plot Theta
axes('position',[pleft+0*(pwid+pspace) pbot+1*(pheight+pspace) pwid pheight])
hold on
lh(1) = plot(1:num_hours,rmse_pred_SW,'Color',mycolors(1,:));
plot(1:num_hours,rmse_ww3_SW*ones(1,num_hours),'--','Color',mycolors(1,:))
lh(2) = plot(1:num_hours,rmse_pred_SS,'Color',mycolors(2,:));
plot(1:num_hours,rmse_ww3_SS*ones(1,num_hours),'--','Color',mycolors(2,:))
%lh = legend(lh,'Swell','Seas');
%set(lh,'box','off')
ylabel('\theta_m RMSE [^o]')
%xlabel('Forecast hour')
set(gca,'XTickLabel',[])
ylim(myylim)
set(gca,'YTick',0:5:10)
grid on
box on

% Plot tm
axes('position',[pleft+0*(pwid+pspace) pbot+0*(pheight+pspace) pwid pheight])
hold on
% Swell
plot([1 24],[tm_sw_rmse_ww3 tm_sw_rmse_ww3],'--','Color',mycolors(1,:))
plot(tm_sw_rmse_pred,'-','Color',mycolors(1,:))
% Seas
plot([1 24],[tm_ss_rmse_ww3 tm_ss_rmse_ww3],'--','Color',mycolors(2,:))
plot(hs_ss_rmse_pred,'-','Color',mycolors(2,:))
%set(gca,'XTickLabel',[])
%set(gca,'YTickLabel',[])
grid on
box on
xlabel('Forecast hour')
ylabel('T_m RMSE [sec]')
ylim(myylimtm)

% SECOND FOLDER
[hs_sw_rmse_ww3, hs_sw_rmse_ww3_br, hs_sw_rmse_pred, hs_ss_rmse_ww3, ...
    hs_ss_rmse_ww3_br, hs_ss_rmse_pred, rmse_pred_SW, rmse_ww3_SW, ...
    rmse_pred_SS, rmse_ww3_SS, tm_sw_rmse_pred, tm_ss_rmse_pred, ...
    tm_sw_rmse_ww3, tm_ss_rmse_ww3] = get_rmse( fol2 );

% Plot HS
axes('position',[pleft+1*(pwid+pspace) pbot+2*(pheight+pspace) pwid pheight])
hold on
% Swell
plot([1 24],[hs_sw_rmse_ww3 hs_sw_rmse_ww3],'--','Color',mycolors(1,:))
plot([1 24],[hs_sw_rmse_ww3_br hs_sw_rmse_ww3_br],'-.','Color',mycolors(1,:))
lh(1)=plot(hs_sw_rmse_pred,'-','Color',mycolors(1,:));
% Seas
plot([1 24],[hs_ss_rmse_ww3 hs_ss_rmse_ww3],'--','Color',mycolors(2,:))
plot([1 24],[hs_ss_rmse_ww3_br hs_ss_rmse_ww3_br],'-.','Color',mycolors(2,:))
lh(2)=plot(hs_ss_rmse_pred,'-','Color',mycolors(2,:));
set(gca,'XTickLabel',[])
set(gca,'YTickLabel',[])
grid on
box on
xlabel('Forecast hour')
ylim(myylimhs)
lha = legend(lh,'Swell','Seas','Location','SouthWest');
set(lha,'box','off')


% Plot Theta
axes('position',[pleft+1*(pwid+pspace) pbot+1*(pheight+pspace) pwid pheight])
hold on
lh(1) = plot(1:num_hours,rmse_pred_SW,'Color',mycolors(1,:));
plot(1:num_hours,rmse_ww3_SW*ones(1,num_hours),'--','Color',mycolors(1,:))
lh(2) = plot(1:num_hours,rmse_pred_SS,'Color',mycolors(2,:));
plot(1:num_hours,rmse_ww3_SS*ones(1,num_hours),'--','Color',mycolors(2,:))
set(gca,'YTickLabel',[])
set(gca,'XTickLabel',[])
%xlabel('Forecast hour')
ylim(myylim)
set(gca,'YTick',0:5:10)
grid on
box on

% Plot tm
axes('position',[pleft+1*(pwid+pspace) pbot+0*(pheight+pspace) pwid pheight])
hold on
% Swell
plot([1 24],[tm_sw_rmse_ww3 tm_sw_rmse_ww3],'--','Color',mycolors(1,:))
plot(tm_sw_rmse_pred,'-','Color',mycolors(1,:))
% Seas
plot([1 24],[tm_ss_rmse_ww3 tm_ss_rmse_ww3],'--','Color',mycolors(2,:))
plot(hs_ss_rmse_pred,'-','Color',mycolors(2,:))
set(gca,'YTickLabel',[])
grid on
box on
xlabel('Forecast hour')
ylim(myylimtm)


% Third FOLDER
[hs_sw_rmse_ww3, hs_sw_rmse_ww3_br, hs_sw_rmse_pred, hs_ss_rmse_ww3, ...
    hs_ss_rmse_ww3_br, hs_ss_rmse_pred, rmse_pred_SW, rmse_ww3_SW, ...
    rmse_pred_SS, rmse_ww3_SS, tm_sw_rmse_pred, tm_ss_rmse_pred, ...
    tm_sw_rmse_ww3, tm_ss_rmse_ww3] = get_rmse( fol3 );

% Plot HS
axes('position',[pleft+2*(pwid+pspace) pbot+2*(pheight+pspace) pwid pheight])
hold on
% Swell
plot([1 24],[hs_sw_rmse_ww3 hs_sw_rmse_ww3],'--','Color',mycolors(1,:))
plot([1 24],[hs_sw_rmse_ww3_br hs_sw_rmse_ww3_br],'-.','Color',mycolors(1,:))
plot(hs_sw_rmse_pred,'-','Color',mycolors(1,:))
% Seas
plot([1 24],[hs_ss_rmse_ww3 hs_ss_rmse_ww3],'--','Color',mycolors(2,:))
plot([1 24],[hs_ss_rmse_ww3_br hs_ss_rmse_ww3_br],'-.','Color',mycolors(2,:))
plot(hs_ss_rmse_pred,'-','Color',mycolors(2,:))
set(gca,'XTickLabel',[])
set(gca,'YTickLabel',[])
grid on
box on
xlabel('Forecast hour')
ylim(myylimhs)

% Plot Theta
axes('position',[pleft+2*(pwid+pspace) pbot+1*(pheight+pspace) pwid pheight])
hold on
lh(1) = plot(1:num_hours,rmse_pred_SW,'Color',mycolors(1,:));
plot(1:num_hours,rmse_ww3_SW*ones(1,num_hours),'--','Color',mycolors(1,:))
lh(2) = plot(1:num_hours,rmse_pred_SS,'Color',mycolors(2,:));
plot(1:num_hours,rmse_ww3_SS*ones(1,num_hours),'--','Color',mycolors(2,:))
set(gca,'YTickLabel',[])
set(gca,'XTickLabel',[])
%xlabel('Forecast hour')
ylim(myylim)
set(gca,'YTick',0:5:10)
grid on
box on

% Plot tm
axes('position',[pleft+2*(pwid+pspace) pbot+0*(pheight+pspace) pwid pheight])
hold on
% Swell
plot([1 24],[tm_sw_rmse_ww3 tm_sw_rmse_ww3],'--','Color',mycolors(1,:))
plot(tm_sw_rmse_pred,'-','Color',mycolors(1,:))
% Seas
plot([1 24],[tm_ss_rmse_ww3 tm_ss_rmse_ww3],'--','Color',mycolors(2,:))
plot(hs_ss_rmse_pred,'-','Color',mycolors(2,:))
set(gca,'YTickLabel',[])
grid on
box on
xlabel('Forecast hour')
ylim(myylimtm)

printFig(gcf,'fig_rmse',[8 5],'png')



return


%% Time Series (CHECK)
% fh = 6;
% figure(1)
% clf
% ax(1)=subplot(211)
% hold on
% plot(obsSW.md1)
% plot(ww3SW.md1)
% plot(predSW(fh).md1)
% plot(predSW(12).md1)
% 
% ax(2)=subplot(212)
% hold on
% plot(obs.hsSW)
% plot(ww3.hsSW)
% plot(pred(fh).hsSW)
% plot(pred(12).hsSW)
% linkaxes(ax,'x')
% 
% 
% subplot(212)
% hold on
% plot(obsSS.md1)
% plot(ww3SS.md1)
% plot(predSS(fh).md1)

%%


