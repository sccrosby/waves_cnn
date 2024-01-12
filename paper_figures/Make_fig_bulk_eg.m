% Figure 
clearvars

% Global set
set(0,'defaultaxesfontsize',8)

% set data directory
data_dir = set_data_dir();



% Load data
fol = strcat(data_dir, '46211_20190811014904_dev_test_6-24_2019_08_22_091752');

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

%%

% Set parameters
hstart = 3960+84*24;
flen = 24*6;
look_back = 6;
fstartV = 24:24:24*5;

% Pick a time step example to show continuous forecast improvement
clf
pleft = 0.1;
pright = .05;
pbot = 0.1;
ptop = 0.05;
pspacev = .01;
pspaceh = .06;
pwid = (1-pleft-pright-0*pspaceh);
pheight = (1-ptop-pbot-pspacev)/3;
mycolors = lines(10);

sdate = datenum(2005,12,31,0,0,0);
edate = datenum(2006,1,7);

% Hsig
axes('position',[pleft+0*(pwid+pspaceh) pbot+2*(pheight+pspacev) pwid pheight])
x = obs.time(hstart:hstart+flen);
y1 = obs.hs(hstart:hstart+flen);
y2 = ww3.hs(hstart:hstart+flen);
for fs = 1:length(fstartV)
for fh=1:num_hours
    yp(fs,fh) = pred(fh).hs(hstart+fstartV(fs)+fh);
    xp(fs,fh) = pred(fh).time(hstart+fstartV(fs)+fh);  
end
    xo(fs,:) = obs.time(hstart+fstartV(fs)-6:hstart+fstartV(fs)-1);
    yo(fs,:) = obs.hs(hstart+fstartV(fs)-6:hstart+fstartV(fs)-1);  
end
lhan(1)=plot(x,y1,'ok');

xlim([sdate edate])
text(sdate + 3/24, 2.25, '(a)', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'fontsize', 8);

hold on
lhan(2)=plot(x,y2,'--','Color','k','LineWidth',1.0);
for fs = 1:length(fstartV)
    plot(xo(fs,:),yo(fs,:),'o','MarkerFaceColor',mycolors(fs,:),'Color','k')
    plot(xp(fs,:),yp(fs,:),'Color',mycolors(fs,:),'LineWidth',1.5)    
end
datetick('x','mm/dd/yyyy')
grid on
ylabel('Hs [m]')
legend('Obs','WW3','Location','Northwest')
set(gca,'XTickLabel',[])





% Tm
axes('position',[pleft+0*(pwid+pspaceh) pbot+1*(pheight+pspacev) pwid pheight])
x = obs.time(hstart:hstart+flen);
y1 = obs.Tm(hstart:hstart+flen);
y2 = ww3.Tm(hstart:hstart+flen);
for fs = 1:length(fstartV)
for fh=1:num_hours
    yp(fs,fh) = pred(fh).Tm(hstart+fstartV(fs)+fh);
    xp(fs,fh) = pred(fh).time(hstart+fstartV(fs)+fh);  
end
    xo(fs,:) = obs.time(hstart+fstartV(fs)-6:hstart+fstartV(fs)-1);
    yo(fs,:) = obs.Tm(hstart+fstartV(fs)-6:hstart+fstartV(fs)-1);  
end
lhan(1)=plot(x,y1,'ok');

xlim([sdate edate])
text(sdate + 3/24, 8.25, '(b)', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'fontsize', 8);

hold on
lhan(2)=plot(x,y2,'--','Color','k','LineWidth',1.);
for fs = 1:length(fstartV)
    plot(xo(fs,:),yo(fs,:),'o','MarkerFaceColor',mycolors(fs,:),'Color','k')
    plot(xp(fs,:),yp(fs,:),'Color',mycolors(fs,:),'LineWidth',1.5)    
end
datetick('x','mm/dd/yyyy')
grid on
ylabel('Tm [sec]')
set(gca,'XTickLabel',[])
set(gca,'YTick',8:2:14)

% md1A
axes('position',[pleft+0*(pwid+pspaceh) pbot+0*(pheight+pspacev) pwid pheight])
x = obs.time(hstart:hstart+flen);
y1 = obs.md1A(hstart:hstart+flen);
y2 = ww3.md1A(hstart:hstart+flen);
for fs = 1:length(fstartV)
for fh=1:num_hours
    yp(fs,fh) = pred(fh).md1A(hstart+fstartV(fs)+fh);
    xp(fs,fh) = pred(fh).time(hstart+fstartV(fs)+fh);  
end
    xo(fs,:) = obs.time(hstart+fstartV(fs)-6:hstart+fstartV(fs)-1);
    yo(fs,:) = obs.md1A(hstart+fstartV(fs)-6:hstart+fstartV(fs)-1);  
end
lhan(1)=plot(x,y1,'ok');

xlim([sdate edate])
text(sdate + 3/24, 212, '(c)', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'fontsize', 8);

hold on
lhan(2)=plot(x,y2,'--','Color','k','LineWidth',1.);
for fs = 1:length(fstartV)
    plot(xo(fs,:),yo(fs,:),'o','MarkerFaceColor',mycolors(fs,:),'Color','k')
    plot(xp(fs,:),yp(fs,:),'Color',mycolors(fs,:),'LineWidth',1.5)    
end
datetick('x','mm/dd/yyyy')
grid on
ylabel('\theta_m [^o]')
set(gca,'XTickLabelRotation',35)
set(gca,'YTick',210:20:280)

printFig(gcf,'fig_03_bulk_eg1',[6.5 6],'pdf')
