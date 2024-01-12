clearvars
addpath D:/Functions_Matlab/

% Load buoy locs
fname = '../met_buoy_obs/noaa_buoys.txt';
fid = fopen(fname,'r');
data = textscan(fid,'%s %f %f');
fclose(fid);
B.id = data{1};
B.lat = data{2};
B.lon = data{3};

% Get Grid
ncfile = 'f:\ww3_cfrs_grid\multi_reanal.partition.wc_4m.197901.nc';
N = ncinfo(ncfile);
W.lat = ncread(ncfile,'latitude');
W.lon = ncread(ncfile,'longitude');
W.lon = W.lon-360;
[W.Lon W.Lat] = meshgrid(W.lon,W.lat);

% Determine nearest cells
for ii = 1:length(B.id)
    [ix(ii) iy(ii)] = findNearestGridPoint(W.Lon,W.Lat,B.lon(ii),B.lat(ii));
end

clf
hold on
plot(W.Lon,W.Lat,'.k')
plot(B.lon,B.lat,'*r')

%%

% Loop over and extract for each location
yr = 1979:2009;
mo = 1:12;
O.wndspd = [];
O.wnddir = [];
O.time = [];
tic
for yy = 1:length(yr)
    for mm = 1:length(mo)
        ncfile = sprintf('f:/ww3_cfrs_grid/multi_reanal.partition.wc_4m.%04d%02d.nc',yr(yy),mo(mm));
        idate = ncread(ncfile,'date');
        idate = datenum(1970,1,1+idate);
        Nt = length(idate);
        iwndspd = NaN(Nt,length(B.id));
        iwnddir = NaN(Nt,length(B.id));
        for ii = 1:length(B.id)
            iwndspd(:,ii) = squeeze(ncread(ncfile,'wind_speed',[iy(ii) ix(ii) 1],[1 1 Inf]));
            iwnddir(:,ii) = squeeze(ncread(ncfile,'wind_direction',[iy(ii) ix(ii) 1],[1 1 Inf]));
        end
        O.wndspd = cat(1,O.wndspd,iwndspd);
        O.wnddir = cat(1,O.wnddir,iwnddir);
        O.time = cat(1,O.time,idate);
    end
    disp(yy)
end
toc

O.lat = B.lat;
O.lon = B.lon;
O.id = B.id;
save('cfsr_buoy_met_pred.mat','-struct','O')

return
