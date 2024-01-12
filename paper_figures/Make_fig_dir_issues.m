% Fdata_dir = set_data_dir();

% Load data
clearvars

% Global set
set(0,'defaultaxesfontsize',8)
data_dir = set_data_dir();

% Load data
fol = strcat(data_dir, '46211_20190811014904_dev_test_6-24_2019_08_22_091752');


files=dir(fol);
fname = files(4).name(1:17);
num_hours = 24;
[ obs, ww3, pred] = load_data( fol, fname, num_hours );

% Plot limits
date_start = datenum(2006,3,1);
date_end = datenum(2006,6,1);

% POST PROCESS: Replace out-of-bounds moments of pred with WW3 
% Consider moving all parameters to WW3 if one is not in range.
tic
for fh = 1:num_hours
    pred(fh).a1F = pred(fh).a1;
    pred(fh).b1F = pred(fh).b1;
    pred(fh).a2F = pred(fh).a2;
    pred(fh).b2F = pred(fh).b2;
    for ff = 1:length(pred(fh).fr)
        inds = abs(pred(fh).a1(:,ff)./pred(fh).e(:,ff)) > 1;        
        pred(fh).a1F(inds,ff) = ww3.a1(inds,ff);
        
        inds = abs(pred(fh).b1(:,ff)./pred(fh).e(:,ff)) > 1;
        pred(fh).b1F(inds,ff) = ww3.b1(inds,ff);
        
        inds = abs(pred(fh).a2(:,ff)./pred(fh).e(:,ff)) > 1;      
        pred(fh).a2F(inds,ff) = ww3.a2(inds,ff);
        
        inds = abs(pred(fh).b2(:,ff)./pred(fh).e(:,ff)) > 1;       
        pred(fh).b2F(inds,ff) = ww3.b2(inds,ff);
    end
end
toc

% Estimate directional integrated for a freq band
abnd = 1:13; %0.04-0.105 Hz
[obs.md1SW,obs.md2SW,obs.spr1SW,obs.spr2SW,~,~]=getkuikstats(sum(obs.a1(:,abnd),2)./sum(obs.e(:,abnd),2),sum(obs.b1(:,abnd),2)./sum(obs.e(:,abnd),2),sum(obs.a2(:,abnd),2)./sum(obs.e(:,abnd),2),sum(obs.b2(:,abnd),2)./sum(obs.e(:,abnd),2));
[ww3.md1SW,ww3.md2SW,ww3.spr1SW,ww3.spr2SW,~,~]=getkuikstats(sum(ww3.a1(:,abnd),2)./sum(ww3.e(:,abnd),2),sum(ww3.b1(:,abnd),2)./sum(ww3.e(:,abnd),2),sum(ww3.a2(:,abnd),2)./sum(ww3.e(:,abnd),2),sum(ww3.b2(:,abnd),2)./sum(ww3.e(:,abnd),2));
for hr = 1:num_hours
    [pred(hr).md1SW,pred(hr).md2SW,pred(hr).spr1SW,pred(hr).spr2SW,~,~]=getkuikstats(sum(pred(hr).a1(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).b1(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).a2(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).b2(:,abnd),2)./sum(pred(hr).e(:,abnd),2));
    [pred(hr).md1SWF,pred(hr).md2SWF,pred(hr).spr1SWF,pred(hr).spr2SWF,~,~]=getkuikstats(sum(pred(hr).a1F(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).b1F(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).a2F(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).b2F(:,abnd),2)./sum(pred(hr).e(:,abnd),2));
end
abnd = 14:28; %0.105 - 0.25 Hz
[obs.md1SS,obs.md2SS,obs.spr1SS,obs.spr2SS,~,~]=getkuikstats(sum(obs.a1(:,abnd),2)./sum(obs.e(:,abnd),2),sum(obs.b1(:,abnd),2)./sum(obs.e(:,abnd),2),sum(obs.a2(:,abnd),2)./sum(obs.e(:,abnd),2),sum(obs.b2(:,abnd),2)./sum(obs.e(:,abnd),2));
[ww3.md1SS,ww3.md2SS,ww3.spr1SS,ww3.spr2SS,~,~]=getkuikstats(sum(ww3.a1(:,abnd),2)./sum(ww3.e(:,abnd),2),sum(ww3.b1(:,abnd),2)./sum(ww3.e(:,abnd),2),sum(ww3.a2(:,abnd),2)./sum(ww3.e(:,abnd),2),sum(ww3.b2(:,abnd),2)./sum(ww3.e(:,abnd),2));
for hr = 1:num_hours
    [pred(hr).md1SS,pred(hr).md2SS,pred(hr).spr1SS,pred(hr).spr2SS,~,~]=getkuikstats(sum(pred(hr).a1(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).b1(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).a2(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).b2(:,abnd),2)./sum(pred(hr).e(:,abnd),2));
    [pred(hr).md1SSF,pred(hr).md2SSF,pred(hr).spr1SSF,pred(hr).spr2SSF,~,~]=getkuikstats(sum(pred(hr).a1F(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).b1F(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).a2F(:,abnd),2)./sum(pred(hr).e(:,abnd),2),sum(pred(hr).b2F(:,abnd),2)./sum(pred(hr).e(:,abnd),2));
end

% Make Plot
clf
pleft = 0.1;
pright = .05;
pbot = 0.1;
ptop = 0.05;
pspace = .01;
pwid = 1-pleft-pright;
pheight = (1-ptop-pbot-pspace)/2;

axes('position',[pleft pbot+1*(pheight+pspace) pwid pheight])
hold on
plot(obs.time,obs.md1SW,'k.')
plot(ww3.time,ww3.md1SW)
fh = 6;
plot(pred(fh).time,pred(fh).md1SW)
plot(pred(fh).time,pred(fh).md1SWF)
% fh = 12;
% plot(pred(fh).time,pred(fh).md1SW)
xlim([date_start date_end])
datetick('x','keeplimits')
set(gca,'XTickLabel',[])
grid on
box on
ylabel('Mean Direction [deg]')


axes('position',[pleft pbot+0*(pheight+pspace) pwid pheight])
hold on
plot(obs.time,obs.md1SS,'k.')
plot(ww3.time,ww3.md1SS)
fh = 6;
plot(pred(fh).time,pred(fh).md1SS)
plot(pred(fh).time,pred(fh).md1SSF)
% fh = 12;
% plot(pred(fh).time,pred(fh).md1SS)
xlim([date_start date_end])
datetick('x','mm/dd/yyyy','keeplimits')
%set(gca,'XTickLabel',[])
grid on
box on
ylabel('Mean Direction [deg]')
legend('Observations','WW3','Adj-6hour','Adj-6hour PP','Location','SouthWest')

printFig(gcf,'fig_dir_issues',[10 5],'png')




