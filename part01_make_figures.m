% Plot results from CNN
% S Crosby, 3/7/19
clearvars

addpath cbrewer\

% Set bounds for time series plots (Later loop for a couple different months during winters)
date_start = datenum(2004,12,1);
date_end = datenum(2005,2,1);

% fol = '2019_03_06_194917_500ep';
% fname = '2019_03_06_200452';
% num_hours = 12;
% [ obs, ww3, pred] = load_data( fol, fname, num_hours );
% 
% fol = 'predictions_20190328215551_relu_e_dev';
% fname = '2019_04_10_163207';
% num_hours = 24;
% [ obs, ww3, pred] = load_data( fol, fname, num_hours );

% fol = 'predictions_46211_20190415200306_dev';
% fname = '2019_04_16_105411';
% num_hours = 24;
% [ obs, ww3, pred] = load_data( fol, fname, num_hours );

% 
% fol = 'predictions_46211_20190419115528_dev_hardtanh';
% fname = '2019_04_19_145502';
% num_hours = 24;
% [ obs, ww3, pred] = load_data( fol, fname, num_hours );

fol = 'C:\Users\Owner\Desktop\noah_research\GraysHarbor\b46211\46211_20211106172742_dev_dev_12-24_2021_11_17_205421';
fname = '2021_11_17_205421';
num_hours = 24;
[ obs, ww3, pred] = load_data( fol, fname, num_hours );
return

%% Plot energy by frequency and forecast hour

% Indices of frequencies to plot
fr_list = [3  8 15]; % 0.0500    0.0750    0.1200 Hz

clf
for ii = 1:length(fr_list)
    % Plot HOUR 1
    hr = 1;
    subplot(3,4,4*(ii-1)+1)
    plot(obs.time,obs.e(:,fr_list(ii)))
    hold on
    plot(ww3.time,ww3.e(:,fr_list(ii)))
    plot(pred(hr).time,pred(hr).e(:,fr_list(ii)))
    legend('Obs','WW3','Pred')
    ylabel(sprintf('Energy at %4.3fHz',obs.fr(fr_list(ii))))
    xlim([date_start date_end])
    datetick('x','mm/yyyy','keeplimits')
    if ii ==1
        title(sprintf('Prediction Hour = %d',hr))
        legend('obs','pred','ww3')
    end
    
    % Plot Hour 6
    hr = 6;
    subplot(3,4,4*(ii-1)+2)
    plot(obs.time,obs.e(:,fr_list(ii)))
    hold on
    plot(ww3.time,ww3.e(:,fr_list(ii)))
    plot(pred(hr).time,pred(hr).e(:,fr_list(ii)))
    legend('Obs','WW3','Pred')
    ylabel(sprintf('Energy at %4.3fHz',obs.fr(fr_list(ii))))
    xlim([date_start date_end])
    datetick('x','mm/yyyy','keeplimits')
    if ii ==1
        title(sprintf('Prediction Hour = %d',hr))
    end
    
    % Plot Hour 12
    hr = 12;
    subplot(3,4,4*(ii-1)+3)
    plot(obs.time,obs.e(:,fr_list(ii)))
    hold on
    plot(ww3.time,ww3.e(:,fr_list(ii)))
    plot(pred(hr).time,pred(hr).e(:,fr_list(ii)))
    legend('Obs','WW3','Pred')
    ylabel(sprintf('Energy at %4.3fHz',obs.fr(fr_list(ii))))
    xlim([date_start date_end])
    datetick('x','mm/yyyy','keeplimits')
    if ii ==1
        title(sprintf('Prediction Hour = %d',hr))
    end
    
    % Plot Hour 12
    hr = 24;
    subplot(3,4,4*ii)
    plot(obs.time,obs.e(:,fr_list(ii)))
    hold on
    plot(ww3.time,ww3.e(:,fr_list(ii)))
    plot(pred(hr).time,pred(hr).e(:,fr_list(ii)))
    legend('Obs','WW3','Pred')
    ylabel(sprintf('Energy at %4.3fHz',obs.fr(fr_list(ii))))
    xlim([date_start date_end])
    datetick('x','mm/yyyy','keeplimits')
    if ii ==1
        title(sprintf('Prediction Hour = %d',hr))
    end
    
end

printFig(gcf,'TimeSeries_EnergyByFreq',[15 8],'pdf')

%% Plot Scatter Plots of Energy by fr and hr

clf
for ii = 1:length(fr_list)
    % Plot HOUR 1
    
    subplot(3,4,4*(ii-1)+1)
    plot(obs.e(:,fr_list(ii)),ww3.e(:,fr_list(ii)),'.')
    hold on
    grid on
    maxE = max([obs.e(:,fr_list(ii)); pred(hr).e(:,fr_list(ii))])
    minE = min([obs.e(:,fr_list(ii)); pred(hr).e(:,fr_list(ii))])
    plot([minE maxE],[minE maxE],'-k')
    xlim([minE maxE])
    ylim([minE maxE])
    ylabel(sprintf('E at %4.3fHz',obs.fr(fr_list(ii))))
    xlabel('Obs E')
    if ii == 1
        title('WW3')
    end

    hr = 1;
    subplot(3,4,4*(ii-1)+2)
    plot(obs.e(:,fr_list(ii)),pred(hr).e(:,fr_list(ii)),'.')
    hold on
    grid on
    maxE = max([obs.e(:,fr_list(ii)); pred(hr).e(:,fr_list(ii))])
    minE = min([obs.e(:,fr_list(ii)); pred(hr).e(:,fr_list(ii))])
    plot([minE maxE],[minE maxE],'-k')
    xlim([minE maxE])
    ylim([minE maxE])
    ylabel(sprintf('E at %4.3fHz',obs.fr(fr_list(ii))))
    xlabel('Obs E')
    if ii == 1
        title(sprintf('Prediction, Hour = %d',hr))
    end
    
    hr = 12;
    subplot(3,4,4*(ii-1)+3)
    plot(obs.e(:,fr_list(ii)),pred(hr).e(:,fr_list(ii)),'.')
    hold on
    grid on
    maxE = max([obs.e(:,fr_list(ii)); pred(hr).e(:,fr_list(ii))])
    minE = min([obs.e(:,fr_list(ii)); pred(hr).e(:,fr_list(ii))])
    plot([minE maxE],[minE maxE],'-k')
    xlim([minE maxE])
    ylim([minE maxE])
    ylabel(sprintf('E at %4.3fHz',obs.fr(fr_list(ii))))
    xlabel('Obs E')
    if ii == 1
        title(sprintf('Prediction, Hour = %d',hr))
    end

    hr = 24;
    subplot(3,4,4*(ii-1)+4)
    plot(obs.e(:,fr_list(ii)),pred(hr).e(:,fr_list(ii)),'.')
    hold on
    grid on
    maxE = max([obs.e(:,fr_list(ii)); pred(hr).e(:,fr_list(ii))])
    minE = min([obs.e(:,fr_list(ii)); pred(hr).e(:,fr_list(ii))])
    plot([minE maxE],[minE maxE],'-k')
    xlim([minE maxE])
    ylim([minE maxE])
    ylabel(sprintf('E at %4.3fHz',obs.fr(fr_list(ii))))
    xlabel('Obs E')
    if ii == 1
        title(sprintf('Prediction, Hour = %d',hr))
    end

end

printFig(gcf,'Scatter_EnergyByFreq',[15 9],'png')



%% Estimate RMSE across freq/hour axes
for ff = 1:length(obs.fr)
    for hr = 1:num_hours
        rmse_pred(hr,ff) = sqrt(nanmean((pred(hr).e(:,ff)-obs.e(:,ff)).^2));
    end
    rmse_ww3(ff) = sqrt(nanmean((ww3.e(:,ff)-obs.e(:,ff)).^2));
end

% Plot linear absolute rmse
mycolors=jet(num_hours);
clf
plot(obs.fr,rmse_ww3,'-k','LineWidth',2)
hold on
for hr = 1:num_hours
    plot(obs.fr,rmse_pred(hr,:),'Color',mycolors(hr,:))
end
grid on
ylabel('RMSE [m^2/Hz]')
xlabel('Frequency [Hz]')
legend('WW3')
chan=colorbar;
colormap(mycolors)
set(chan,'Xtick',1/num_hours*2:1/num_hours:1,'XtickLabel',1:num_hours)
ylabel(chan,'Forecast hour')
printFig(gcf,'RMSE_Absolute_linear',[8 6],'pdf')


%% Estimate relative RMSE loss/gain
for ff = 1:length(obs.fr)
    for hr = 1:num_hours
        rmse_loss(hr,ff) = (rmse_pred(hr,ff)-rmse_ww3(ff))/rmse_ww3(ff);
    end
end

% Custom colormap using cbrewer (Red-Blue with white in middle)
addpath cbrewer
mycolors = flipud(cbrewer('div','RdBu',101));

% Plot color gain
clf
ax = contourf(obs.fr,1:num_hours,rmse_loss,20,'EdgeColor','none');
shading flat
chan = colorbar('Location','NorthOutside');
colormap(mycolors)
caxis([-.5 .5])
ylabel('Forecast time [Hr]')
xlabel('Frequency [Hz]')
ylabel(chan,'RMSE Change [%]')
printFig(gcf,'RMSE_Loss_relative',[8 8],'pdf')



%% Integrate Bulk Hs, Tm, md1 across freq (note Hs becomes imaginary when e < 0 ),

% Integrate for Hs (Significant Wave Height)
ww3.hs = 4*sqrt(ww3.e*ww3.bw');
obs.hs = 4*sqrt(obs.e*obs.bw');
for hr = 1:num_hours
    pred(hr).hs = 4*sqrt(pred(hr).e*pred(hr).bw');
end

% Calc Tm (mean wave period)
%   First calculate mean freq, this is a weight average at each time step
%   of fr and e. Then Tm = 1/fm
ww3.fm = sum(ww3.e.*repmat(ww3.fr,[length(ww3.time) 1]),2)./sum(ww3.e,2);
ww3.Tm = 1./ww3.fm;
obs.fm = sum(obs.e.*repmat(obs.fr,[length(obs.time) 1]),2)./sum(obs.e,2);
obs.Tm = 1./obs.fm;
for hr = 1:num_hours
    pred(hr).fm = sum(pred(hr).e.*repmat(pred(hr).fr,[length(pred(hr).time) 1]),2)./sum(pred(hr).e,2);
    pred(hr).Tm = 1./pred(hr).fm;
end

% Estimate mean wave direction (direction wave goes)
[obs.md1A,obs.md2A,obs.spr1A,obs.spr2A,~,~]=getkuikstats(sum(obs.a1,2)./sum(obs.e,2),sum(obs.b1,2)./sum(obs.e,2),sum(obs.a2,2)./sum(obs.e,2),sum(obs.b2,2)./sum(obs.e,2));
[ww3.md1A,ww3.md2A,ww3.spr1A,ww3.spr2A,~,~]=getkuikstats(sum(ww3.a1,2)./sum(ww3.e,2),sum(ww3.b1,2)./sum(ww3.e,2),sum(ww3.a2,2)./sum(ww3.e,2),sum(ww3.b2,2)./sum(ww3.e,2));
for hr = 1:num_hours
    %Interested in md1A
    [pred(hr).md1A,pred(hr).md2A,pred(hr).spr1A,pred(hr).spr2A,~,~]=getkuikstats(sum(pred(hr).a1,2)./sum(pred(hr).e,2),sum(pred(hr).b1,2)./sum(pred(hr).e,2),sum(pred(hr).a2,2)./sum(pred(hr).e,2),sum(pred(hr).b2,2)./sum(pred(hr).e,2));
end

%rmse(hr_train, hr_for) = sqrt(mean(pred(hr_for).Hs - obs
clf
ii = 0;
for hr = [1 6 12 24]
    ii = ii +1;
    
    % Hs
    subplot(4,3,(ii-1)*3+1)
    hold on
    plot(obs.time,obs.hs)    
    plot(ww3.time,ww3.hs)
    plot(pred(hr).time,pred(hr).hs)
    legend('Obs','WW3','Pred')
    ylabel('Hs [m]')
    xlim([date_start date_end])
    datetick('x','mm/yyyy','keeplimits')
    grid on
    box on
    title(sprintf('Prediction Hour = %d',hr))
    
    % Tm
    subplot(4,3,(ii-1)*3+2)
    hold on
    plot(obs.time,obs.Tm)    
    plot(ww3.time,ww3.Tm)
    plot(pred(hr).time,pred(hr).Tm)
    ylabel('Tm [sec]')
    ylim([0 18])
    xlim([date_start date_end])
    datetick('x','mm/yyyy','keeplimits')
    grid on
    box on
    title(sprintf('Prediction Hour = %d',hr))
    
    % Md1
    subplot(4,3,(ii-1)*3+3)
    hold on
    plot(obs.time,obs.md1A)    
    plot(ww3.time,ww3.md1A)
    plot(pred(hr).time,pred(hr).md1A)
    %legend('Obs','Pred','WW3')
    ylabel('Mean Direction [deg]')
    xlim([date_start date_end])
    datetick('x','mm/yyyy','keeplimits')
    grid on
    box on
    title(sprintf('Prediction Hour = %d',hr))
end

printFig(gcf,'TimeSeries_BulkHsTmMd1',[15 8],'pdf')

%% RMSE by Wave Height

for hr = 1:num_hours
    rmse_hs(hr) = real(sqrt(mean(nanmean((pred(hr).hs-obs.hs).^2))));
end

rmse_ww3 = sqrt(mean(nanmean((ww3.hs-obs.hs).^2)));

clf
plot(rmse_hs)
hold on
plot([0 25],[rmse_ww3 rmse_ww3],'-r')
grid on
ylim([0 .35])
ylabel('RMSE [m]')
xlabel('Forecast Hour')
legend('Pred','WW3')

printFig(gcf,'Hs_RMSE',[7 5],'pdf')

%% Estimate directional components by frequency

[obs.md1,obs.md2,obs.spr1,obs.spr2,~,~]=getkuikstats(obs.a1./obs.e,obs.b1./obs.e,obs.a2./obs.e,obs.b2./obs.e);
[ww3.md1,ww3.md2,ww3.spr1,ww3.spr2,~,~]=getkuikstats(ww3.a1./ww3.e,ww3.b1./ww3.e,ww3.a2./ww3.e,ww3.b2./ww3.e);
for hr = 1:12
    [pred(hr).md1,pred(hr).md2,pred(hr).spr1,pred(hr).spr2,~,~]=getkuikstats(pred(hr).a1./pred(hr).e,pred(hr).b1./pred(hr).e,pred(hr).a2./pred(hr).e,pred(hr).b2./pred(hr).e);
end

clf
% Indices of frequencies to plot
fr_list = [3  8 15];
for ii = 1:length(fr_list)
    % Plot HOUR 1
    hr = 1;
    subplot(3,3,3*(ii-1)+1)
    plot(obs.time,obs.md1(:,fr_list(ii)))
    hold on    
    plot(ww3.time,ww3.md1(:,fr_list(ii)))
    plot(pred(hr).time,pred(hr).md1(:,fr_list(ii)))
    legend('Obs','WW3','Pred')
    ylabel(sprintf('Energy at %4.3fHz',obs.fr(fr_list(ii))))
    xlim([date_start date_end])
    datetick('x','mm/yyyy','keeplimits')
    if ii ==1
        title(sprintf('Prediction Hour = %d',hr))
        legend('obs','pred','ww3')
    end
    
    % Plot Hour 2
    hr = 6;
    subplot(3,3,3*(ii-1)+2)
    plot(obs.time,obs.md1(:,fr_list(ii)))
    hold on    
    plot(ww3.time,ww3.md1(:,fr_list(ii)))
    plot(pred(hr).time,pred(hr).md1(:,fr_list(ii)))
    legend('Obs','Pred','WW3')
    ylabel(sprintf('Energy at %4.3fHz',obs.fr(fr_list(ii))))
    xlim([date_start date_end])
    datetick('x','mm/yyyy','keeplimits')
    if ii ==1
        title(sprintf('Prediction Hour = %d',hr))
    end
    
    % Plot Hour 2
    hr = 12;
    subplot(3,3,3*ii)
    plot(obs.time,obs.md1(:,fr_list(ii)))
    hold on    
    plot(ww3.time,ww3.md1(:,fr_list(ii)))
    plot(pred(hr).time,pred(hr).md1(:,fr_list(ii)))
    legend('Obs','Pred','WW3')
    ylabel(sprintf('Energy at %4.3fHz',obs.fr(fr_list(ii))))
    xlim([date_start date_end])
    datetick('x','mm/yyyy','keeplimits')
    if ii ==1
        title(sprintf('Prediction Hour = %d',hr))
    end
    
    
end

printFig(gcf,'TimeSeries_DirByFreq',[15 8],'pdf')


















