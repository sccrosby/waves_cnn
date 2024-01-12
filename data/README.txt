Description of files
Obs and Predictions at NOAA NDBC buoy location 46211: https://www.ndbc.noaa.gov/station_page.php?station=46211

Link to data: https://wwu2-my.sharepoint.com/:f:/r/personal/crosbys4_wwu_edu/Documents/WaveForecasting_CS?csf=1&e=6loEDO

Running matlab on cs machines:


WW3CFSRphase2_46211_rebanded.mat
Overview: Wave spectra predictions from WW3 CFSR Phase 2 reanalysis at NOAA buoy site 46211, Gray's Harbor, offshore of WA
Variables
- time (90584x1): Matlab vector datenum format, dt = 3-hour, 1979-2009.
- sp (64x36x90584): Wave energy density spectra, E(freq,theta,time), units: m^2/(Hz-deg)
- dir (36x1): Direction vector for theta, units: deg
- fr (64x1): Frequency vector, units: Hz
- bw  (64x1): Bandwidth, units Hz
- e: Wave energy freq spectra, integrated from sp. units: m^2/Hz
- hs: Significant wave height, units: m
- dtheta (1x1): 10
- lat, lon (1x1): fixed values

036p1_historic.nc
- time (1x216936): Matlab vector datenum format, dt = 1-hour, 1993-Present. However data gaps exist, and e,a1,b1 etc are filled with NaN
- e: Wave energy freq spectra, units: m^2/Hz
- a1 (64x216936) = first order directional moment, a1(f) = int[E(f,theta)*cos(theta) dtheta]
- b1 (64x216936) = second order directional moment, b1(f) = int[E(f,theta)*sin(theta) dtheta]
- a2 (64x216936) = first order directional moment, a1(f) = int[E(f,theta)*cos(2*theta) dtheta]
- b2 (64x216936) = second order directional moment, b1(f) = int[E(f,theta)*sin(2*theta) dtheta]
- fr (64x1): frequency, units: Hz
- bw (64x1): bandwidth, units: Hz
