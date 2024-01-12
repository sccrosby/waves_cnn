clearvars

addpath D:/Functions_Matlab/

K = kml2struct('buoy_locs.kml');

ncfile = 'multi_reanal.partition.wc_4m.197901.nc';
N = ncinfo(ncfile);

W.lat = ncread(ncfile,'latitude');
W.lon = ncread(ncfile,'longitude');
W.lon = W.lon-360;

%%
W.hs = ncread(ncfile,'significant_wave_height',[1 1 1 500],[736 526 19 1]);
W.wspd = ncread(ncfile,'wind_speed',[1 1 500],[736 526 1]);
W.wdir = ncread(ncfile,'wind_direction',[1 1 500],[736 526 1]);

W.u = W.wspd.*cosd(90-W.wdir+180);
W.v = W.wspd.*cosd(90-W.wdir+180);

%% Wave grid
dx = .1;
lat = 41:dx:49.4; %Keep 49.4 fixed)
lon = -126:dx:-123.7;
[Lat, Lon] = meshgrid(lat,lon);


clf
subplot(121)
imagesc(W.lon,W.lat,W.hs(:,:,1)')
shading flat
colorbar
hold on
set(gca,'YDir','normal')
plot(Lon,Lat,'k.')
plot([K.Lon],[K.Lat],'r*')
xlim([-127 -122])
ylim([39 50])

subplot(122)
pcolor(W.lon,W.lat,W.wspd')
hold on
quiver(W.lon,W.lat,W.u,W.v)
shading flat
xlim([-127 -122])
ylim([39 50])
colorbar

%% West boundary
Ix = find(W.lon==lon(1));
Iy = find(W.lat==lat(1)):517;

ii = 1;
P.hs = squeeze(ncread(ncfile,'significant_wave_height',[Ix Iy(ii) 1 1],[1 1 19 745]));
P.tp = squeeze(ncread(ncfile,'peak_period',[Ix Iy(ii) 1 1],[1 1 19 745]));
P.md = squeeze(ncread(ncfile,'wave_direction',[Ix Iy(ii) 1 1],[1 1 19 745]));
P.dspr = squeeze(ncread(ncfile,'direction_spreading',[Ix Iy(ii) 1 1],[1 1 19 745]));
P.wsfrac = squeeze(ncread(ncfile,'wind_sea_fraction',[Ix Iy(ii) 1 1],[1 1 19 745]));

%%

jon_gam = 3.3;
f = .035:.005:1;
bw = diff(f(1:2));
theta = 0.5:359.5; %Do not change
dtheta = pi/180;

tt = 400;
cc = 1;
E = zeros(length(f),length(theta));
while cc <= 19    
hs = P.hs(cc,tt);
tp = P.tp(cc,tt);
dspr = P.dspr(cc,tt)*pi/180;
md = P.md(cc,tt);
if ~isnan(hs)
    [Ei, ei]= reconstruct_2d_spectra(hs,tp,dspr,md,jon_gam,f,theta,bw,dtheta);
    if sum(isnan(Ei(:))) > 0
        return
    end
    E = E+Ei;
    cc = cc + 1;
else
    break
end

end

clf
subplot(2,2,1)
plot(ei,f)
set(gca,'XDir','reverse')

subplot(2,2,2)
pcolor(theta,f,log10(E))
shading flat
colorbar
caxis([-3 2])

subplot(2,2,4)
semilogy(theta,sum(E,1))
ylim([1e-4 1000])

% Check
hs_check = 4*sqrt(sum(E(:))*dtheta*bw);













