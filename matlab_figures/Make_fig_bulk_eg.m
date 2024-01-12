% Figure 
clearvars

% Global set
set(0,'defaultaxesfontsize',8)

% Load data
fol = '46211_20190601094513_dev_dev_24-24_2019_06_01_102532';

% built-in for jonny to run this on a linux machine
if isunix && strfind(pwd, 'jonny')
    fol = strcat('/home/hutch_research/projects/ml_waves19_jonny/data_outputs/',fol);
    addpath /home/hutch_research/projects/ml_waves19/MatlabPlottingCode/cbrewer
end

files=dir(fol);
fname = files(4).name(1:17);

num_hours = 24;
[ obs, ww3, pred] = load_data( fol, fname, num_hours );

% Integrate for Hs
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

% Estimate directional integrated across frequency
[obs.md1A,obs.md2A,obs.spr1A,obs.spr2A,~,~]=getkuikstats(sum(obs.a1,2)./sum(obs.e,2),sum(obs.b1,2)./sum(obs.e,2),sum(obs.a2,2)./sum(obs.e,2),sum(obs.b2,2)./sum(obs.e,2));
[ww3.md1A,ww3.md2A,ww3.spr1A,ww3.spr2A,~,~]=getkuikstats(sum(ww3.a1,2)./sum(ww3.e,2),sum(ww3.b1,2)./sum(ww3.e,2),sum(ww3.a2,2)./sum(ww3.e,2),sum(ww3.b2,2)./sum(ww3.e,2));
for hr = 1:num_hours
    [pred(hr).md1A,pred(hr).md2A,pred(hr).spr1A,pred(hr).spr2A,~,~]=getkuikstats(sum(pred(hr).a1,2)./sum(pred(hr).e,2),sum(pred(hr).b1,2)./sum(pred(hr).e,2),sum(pred(hr).a2,2)./sum(pred(hr).e,2),sum(pred(hr).b2,2)./sum(pred(hr).e,2));
end


% Pick a time step example to show continuous forecast improvement
clf
pleft = 0.1;
pright = .05;
pbot = 0.1;
ptop = 0.05;
pspacev = .01;
pspaceh = .06;
pwid = (1-pleft-pright-2*pspaceh)/2;
pheight = (1-ptop-pbot-pspacev)/3;

% Example 1
axes('position',[pleft+0*(pwid+pspaceh) pbot+2*(pheight+pspacev) pwid pheight])
fstart = 3960;
hstart = fstart - 36;
flen = 24;
ftime = obs.time(hstart:fstart+flen);
ptime = obs.time(fstart+1:fstart+flen);
ohs = obs.hs(hstart:fstart+flen);
whs = ww3.hs(hstart:fstart+flen);
for fh=1:num_hours
    phs(fh) = pred(fh).hs(fstart+fh);
end
plot(ftime,ohs,'.k')
hold on
plot(ftime,whs)
plot(ptime,phs)
datetick('x','mm/dd/yyyy')
grid on
ylabel('Hs [m]')
legend('Obs','WW3','Adj','Location','Northwest')
set(gca,'XTickLabel',[])

axes('position',[pleft+0*(pwid+pspaceh) pbot+1*(pheight+pspacev) pwid pheight])
ftime = obs.time(hstart:fstart+flen);
ptime = obs.time(fstart+1:fstart+flen);
ohs = obs.Tm(hstart:fstart+flen);
whs = ww3.Tm(hstart:fstart+flen);
for fh=1:num_hours
    phs(fh) = pred(fh).Tm(fstart+fh);
end
plot(ftime,ohs,'.k')
hold on
plot(ftime,whs)
plot(ptime,phs)
datetick('x','mm/dd/yyyy')
grid on
ylabel('Tm [sec]')
set(gca,'XTickLabel',[])

axes('position',[pleft+0*(pwid+pspaceh) pbot+0*(pheight+pspacev) pwid pheight])
ftime = obs.time(hstart:fstart+flen);
ptime = obs.time(fstart+1:fstart+flen);
ohs = obs.md1A(hstart:fstart+flen);
whs = ww3.md1A(hstart:fstart+flen);
for fh=1:num_hours
    phs(fh) = pred(fh).md1A(fstart+fh);
end
plot(ftime,ohs,'.k')
hold on
plot(ftime,whs)
plot(ptime,phs)
datetick('x','mm/dd/yyyy')
grid on
ylabel('\theta_m [^o]')


% Example 2
axes('position',[pleft+1*(pwid+pspaceh) pbot+2*(pheight+pspacev) pwid pheight])
fstart = 2995;
hstart = fstart - 36;
flen = 24;
ftime = obs.time(hstart:fstart+flen);
ptime = obs.time(fstart+1:fstart+flen);
ohs = obs.hs(hstart:fstart+flen);
whs = ww3.hs(hstart:fstart+flen);
for fh=1:num_hours
    phs(fh) = pred(fh).hs(fstart+fh);
end
plot(ftime,ohs,'.k')
hold on
plot(ftime,whs)
plot(ptime,phs)
datetick('x','mm/dd/yyyy')
grid on
ylabel('Hs [m]')
legend('Obs','WW3','Adj','Location','Northwest')
set(gca,'XTickLabel',[])

axes('position',[pleft+1*(pwid+pspaceh) pbot+1*(pheight+pspacev) pwid pheight])
ftime = obs.time(hstart:fstart+flen);
ptime = obs.time(fstart+1:fstart+flen);
ohs = obs.Tm(hstart:fstart+flen);
whs = ww3.Tm(hstart:fstart+flen);
for fh=1:num_hours
    phs(fh) = pred(fh).Tm(fstart+fh);
end
plot(ftime,ohs,'.k')
hold on
plot(ftime,whs)
plot(ptime,phs)
datetick('x','mm/dd/yyyy')
grid on
ylabel('Tm [sec]')
set(gca,'XTickLabel',[])

axes('position',[pleft+1*(pwid+pspaceh) pbot+0*(pheight+pspacev) pwid pheight])
ftime = obs.time(hstart:fstart+flen);
ptime = obs.time(fstart+1:fstart+flen);
ohs = obs.md1A(hstart:fstart+flen);
whs = ww3.md1A(hstart:fstart+flen);
for fh=1:num_hours
    phs(fh) = pred(fh).md1A(fstart+fh);
end
plot(ftime,ohs,'.k')
hold on
plot(ftime,whs)
plot(ptime,phs)
datetick('x','mm/dd/yyyy')
grid on
ylabel('\theta_m [^o]')

printFig(gcf,'fig_bulk_eg',[8 8],'png')








return 
%% Examine a bunch of times
clf
pleft = 0.1;
pright = .05;
pbot = 0.1;
ptop = 0.05;
pspace = .04;
pwid = (1-pleft-pright-2*pspace)/3;
pheight = (1-ptop-pbot-pspace)/2;

axes('position',[pleft+0*(pwid+pspace) pbot+1*(pheight+pspace) pwid pheight])
fstart = 900;
hstart = fstart - 36;
flen = 24;
ftime = obs.time(hstart:fstart+flen);
ptime = obs.time(fstart+1:fstart+flen);
ohs = obs.hs(hstart:fstart+flen);
whs = ww3.hs(hstart:fstart+flen);
for fh=1:num_hours
    phs(fh) = pred(fh).hs(fstart+fh);
end
plot(ftime,ohs,'.k')
hold on
plot(ftime,whs)
plot(ptime,phs)
datetick('x','mm/dd/yyyy')
grid on
ylabel('Hs [m]')
legend('Obs','WW3','Adj')


axes('position',[pleft+1*(pwid+pspace) pbot+1*(pheight+pspace) pwid pheight])
fstart = 1000;
hstart = fstart - 36;
flen = 24;
ftime = obs.time(hstart:fstart+flen);
ptime = obs.time(fstart+1:fstart+flen);
ohs = obs.hs(hstart:fstart+flen);
whs = ww3.hs(hstart:fstart+flen);
for fh=1:num_hours
    phs(fh) = pred(fh).hs(fstart+fh);
end
plot(ftime,ohs,'.k')
hold on
plot(ftime,whs)
plot(ptime,phs)
datetick('x','mm/dd/yyyy')
grid on
ylabel('Hs [m]')

axes('position',[pleft+2*(pwid+pspace) pbot+1*(pheight+pspace) pwid pheight])
fstart = 2000;
hstart = fstart - 36;
flen = 24;
ftime = obs.time(hstart:fstart+flen);
ptime = obs.time(fstart+1:fstart+flen);
ohs = obs.hs(hstart:fstart+flen);
whs = ww3.hs(hstart:fstart+flen);
for fh=1:num_hours
    phs(fh) = pred(fh).hs(fstart+fh);
end
plot(ftime,ohs,'.k')
hold on
plot(ftime,whs)
plot(ptime,phs)
datetick('x','mm/dd/yyyy')
grid on
ylabel('Hs [m]')

axes('position',[pleft+0*(pwid+pspace) pbot+0*(pheight+pspace) pwid pheight])
fstart = 3000;
hstart = fstart - 36;
flen = 24;
ftime = obs.time(hstart:fstart+flen);
ptime = obs.time(fstart+1:fstart+flen);
ohs = obs.hs(hstart:fstart+flen);
whs = ww3.hs(hstart:fstart+flen);
for fh=1:num_hours
    phs(fh) = pred(fh).hs(fstart+fh);
end
plot(ftime,ohs,'.k')
hold on
plot(ftime,whs)
plot(ptime,phs)
datetick('x','mm/dd/yyyy')
grid on
ylabel('Hs [m]')


axes('position',[pleft+1*(pwid+pspace) pbot+0*(pheight+pspace) pwid pheight])
fstart = 3965;
hstart = fstart - 36;
flen = 24;
ftime = obs.time(hstart:fstart+flen);
ptime = obs.time(fstart+1:fstart+flen);
ohs = obs.hs(hstart:fstart+flen);
whs = ww3.hs(hstart:fstart+flen);
for fh=1:num_hours
    phs(fh) = pred(fh).hs(fstart+fh);
end
plot(ftime,ohs,'.k')
hold on
plot(ftime,whs)
plot(ptime,phs)
datetick('x','mm/dd/yyyy')
grid on
ylabel('Hs [m]')


axes('position',[pleft+2*(pwid+pspace) pbot+0*(pheight+pspace) pwid pheight])
fstart = 5000;
hstart = fstart - 36;
flen = 24;
ftime = obs.time(hstart:fstart+flen);
ptime = obs.time(fstart+1:fstart+flen);
ohs = obs.hs(hstart:fstart+flen);
whs = ww3.hs(hstart:fstart+flen);
for fh=1:num_hours
    phs(fh) = pred(fh).hs(fstart+fh);
end
plot(ftime,ohs,'.k')
hold on
plot(ftime,whs)
plot(ptime,phs)
datetick('x','mm/dd/yyyy')
grid on
ylabel('Hs [m]')

