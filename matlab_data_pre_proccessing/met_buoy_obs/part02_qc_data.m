clearvars

load('buoy_bulkwave_met_obs.mat')

% no data here
M([2 7 12 19])=[];
id([2 7 12 19])=[];

for ii = 1:length(M)
M(ii).u = M(ii).wndspd.*cosd(90-M(ii).wnddir+180);
M(ii).v = M(ii).wndspd.*sind(90-M(ii).wnddir+180);
end
%% Interp onto common time, remove outliers
O.time = datenum(1980,1,1):(1/48):datenum(2010,10,1);

for ii = 1:length(M)
    [~, I] = unique(M(ii).time);
    
    % Wave
    O.hs(ii,:) = interpShortGap(M(ii).time(I),M(ii).waveheight(I),O.time,4);
    O.tp(ii,:) = interpShortGap(M(ii).time(I),M(ii).wavepeakperiod(I),O.time,4);
    O.tm(ii,:) = interpShortGap(M(ii).time(I),M(ii).wavemeanperiod(I),O.time,4);
    O.md(ii,:) = interpShortGap(M(ii).time(I),M(ii).wavedir(I),O.time,4);
    
    % Wind
    O.wndspd(ii,:) = interpShortGap(M(ii).time(I),M(ii).wndspd(I),O.time,4);
    O.wndgust(ii,:) = interpShortGap(M(ii).time(I),M(ii).wndgust(I),O.time,4);
    O.u(ii,:) = interpShortGap(M(ii).time(I),M(ii).u(I),O.time,4);
    O.v(ii,:) = interpShortGap(M(ii).time(I),M(ii).v(I),O.time,4);
       
    % Wind QC
    mn = nanmean(O.wndspd(ii,:));
    sd = nanstd(O.wndspd(ii,:));
    rm = O.wndspd(ii,:) > (mn+5*sd);    
    O.wndspd(ii,rm) = NaN;
    O.u(ii,rm) = NaN;
    O.v(ii,rm) = NaN;
end
O.wnddir = wrapTo360(90-atan2d(O.v,O.u)+180);
O.id = id;

save('buoy_bulkwave_met_obs_qc.mat','-struct','O')


%% Quick Plot of wind data

clf
hold on
for ii = 1:length(M)
    plot(O.time,O.wndspd(ii,:) + ii*50,'-k')
    text(O.time(end),ii*50,id{ii})
end
datetick('x')
ylabel('Wind Speed [m/s]')
printFig(gcf,'wind_data',[12 12],'png')
    
%% Quick plot of wave data
clf
hold on
for ii = 1:length(M)
    plot(O.time,O.hs(ii,:) + ii*10,'-k')
    text(O.time(end),ii*10,id{ii})
end
datetick('x')
ylabel('Hs [m]')
printFig(gcf,'wave_data',[12 12],'png')

%% Wind Rose
clf
for ii = 1:21;
ax = subplot(3,7,ii);
[figure_handle,count,speeds,directions,Table] = WindRose(O.wnddir(ii,:),O.wndspd(ii,:),'axes',ax,'legendtype',1,'labels',{'','E','S','W'});
title(id{ii})
end
printFig(gcf,'wind_roses',[20 5],'png')









