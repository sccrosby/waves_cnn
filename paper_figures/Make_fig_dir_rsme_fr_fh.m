% Figure rsme_fr_fh
clearvars
addpath cbrewer

% Global set
set(0,'defaultaxesfontsize',8)

% Pick 3 models to plot
% Dev
fol1 = '46211_20190722174531_dev_dev_6-24_2019_07_23_120706'
fol2 = '46211_20190721220720_dev_dev_12-24_2019_07_23_120908'
fol3 = '46211_20190722072410_dev_dev_24-24_2019_07_23_121019'

% Test
fol1 = '46211_20190722174531_dev_test_6-24_2019_07_23_120733'
fol2 = '46211_20190721220720_dev_test_12-24_2019_07_23_120932'
fol3 = '46211_20190722072410_dev_test_24-24_2019_07_23_121048'

% built-in for jonny to run this on a linux machine
if isunix && strfind(pwd, 'jonny')
    fol1 = strcat('/home/hutch_research/projects/ml_waves19_jonny/data_outputs/',fol1);
    fol2 = strcat('/home/hutch_research/projects/ml_waves19_jonny/data_outputs/',fol2);
    fol3 = strcat('/home/hutch_research/projects/ml_waves19_jonny/data_outputs/',fol3);
    addpath /home/hutch_research/projects/ml_waves19/MatlabPlottingCode/cbrewer
end


% Custom colormap using cbrewer (Red-Blue with white in middle)
mycolors = flipud(cbrewer('div','RdBu',101));

num_hours = 24;
myxlim = [.04 .25];

clf
pleft = .07;
pright = .05;
ptop = .15;
pbot = .1;
pspace = .01;
pwid = (1-pleft-pright-2*pspace)/3;
pheight = 1-ptop-pbot;


%%%%%%%%%%%%%%%%%%%%%%% Load data AND PLOT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files=dir(fol1);
fname = files(4).name(1:17);
[ obs, ww3, pred] = load_data( fol1, fname, num_hours );

% Remove out of bounds directional moments
% pred = removeOutofBounds( pred, ww3, num_hours, 'all' );

% Estimate mean dir
obs = calcMeanDir( obs, 1, length(obs.fr) );
ww3 = calcMeanDir( ww3, 1, length(ww3.fr) );
pred = calcMeanDir( pred, num_hours, length(pred(1).fr) );

% Calc RMSE
for ff = 1:length(obs.fr)
    for hr = 1:num_hours
        temp = pred(hr).md1(:,ff)-obs.md1(:,ff);
        % fix wrapping
        temp(temp>180) = temp(temp>180)-360;
        temp(temp<-180) = temp(temp<-180)+360;
        tempW = obs.e(:,ff);
        rmse_pred(hr,ff) = sqrt(nansum(tempW.*temp.^2)/sum(tempW));
        %bias(hr,ff) = nanmean(
    end
    temp = ww3.md1(:,ff)-obs.md1(:,ff);
    % fix wrapping
    temp(temp>180) = temp(temp>180)-360;
    temp(temp<-180) = temp(temp<-180)+360;
    rmse_ww3(ff) = sqrt(nansum(tempW.*temp.^2)/sum(tempW));
end


%Estimate relative RMSE loss/gain
for ff = 1:length(obs.fr)
    for hr = 1:num_hours
        rmse_change(hr,ff) = (rmse_pred(hr,ff)-rmse_ww3(ff));     
    end
end

axes('position',[pleft+0*(pwid+pspace) pbot pwid pheight])
ax = contourf(obs.fr,1:num_hours,rmse_change,-50:50,'EdgeColor','none');
shading flat
colormap(mycolors)
caxis([-10 10])
grid on
xlim(myxlim)
ylabel('Forecast time [hr]')
xlabel('Frequency [Hz]')


%%%%%%%%%%%%%%%%%%%%%%% Load data AND PLOT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files=dir(fol2);
fname = files(4).name(1:17);
[ obs, ww3, pred] = load_data( fol2, fname, num_hours );

% Estimate mean dir
obs = calcMeanDir( obs, 1, length(obs.fr) );
ww3 = calcMeanDir( ww3, 1, length(ww3.fr) );
pred = calcMeanDir( pred, num_hours, length(pred(1).fr) );

% Calc RMSE
for ff = 1:length(obs.fr)
    for hr = 1:num_hours
        temp = pred(hr).md1(:,ff)-obs.md1(:,ff);
        % fix wrapping
        temp(temp>180) = temp(temp>180)-360;
        temp(temp<-180) = temp(temp<-180)+360;
        tempW = obs.e(:,ff);
        rmse_pred(hr,ff) = sqrt(nansum(tempW.*temp.^2)/sum(tempW));
    end
    temp = ww3.md1(:,ff)-obs.md1(:,ff);
    % fix wrapping
    temp(temp>180) = temp(temp>180)-360;
    temp(temp<-180) = temp(temp<-180)+360;
    rmse_ww3(ff) = sqrt(nansum(tempW.*temp.^2)/sum(tempW));
end


%Estimate relative RMSE loss/gain
for ff = 1:length(obs.fr)
    for hr = 1:num_hours
        rmse_change(hr,ff) = (rmse_pred(hr,ff)-rmse_ww3(ff));     
    end
end

axes('position',[pleft+1*(pwid+pspace) pbot pwid pheight])
ax = contourf(obs.fr,1:num_hours,rmse_change,-50:50,'EdgeColor','none');
shading flat
colormap(mycolors)
caxis([-10 10])
grid on
xlim(myxlim)
xlabel('Frequency [Hz]')
set(gca,'YTickLabel',[])

%%%%%%%%%%%%%%%%%%%%%%% Load data AND PLOT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files=dir(fol3);
fname = files(4).name(1:17);
[ obs, ww3, pred] = load_data( fol3, fname, num_hours );

% Estimate mean dir
obs = calcMeanDir( obs, 1, length(obs.fr) );
ww3 = calcMeanDir( ww3, 1, length(ww3.fr) );
pred = calcMeanDir( pred, num_hours, length(pred(1).fr) );

% Calc RMSE
for ff = 1:length(obs.fr)
    for hr = 1:num_hours
        temp = pred(hr).md1(:,ff)-obs.md1(:,ff);
        % fix wrapping
        temp(temp>180) = temp(temp>180)-360;
        temp(temp<-180) = temp(temp<-180)+360;
        tempW = obs.e(:,ff);
        rmse_pred(hr,ff) = sqrt(nansum(tempW.*temp.^2)/sum(tempW));
    end
    temp = ww3.md1(:,ff)-obs.md1(:,ff);
    % fix wrapping
    temp(temp>180) = temp(temp>180)-360;
    temp(temp<-180) = temp(temp<-180)+360;
    rmse_ww3(ff) = sqrt(nansum(tempW.*temp.^2)/sum(tempW));
end


%Estimate relative RMSE loss/gain
for ff = 1:length(obs.fr)
    for hr = 1:num_hours
        rmse_change(hr,ff) = (rmse_pred(hr,ff)-rmse_ww3(ff));     
    end
end

axes('position',[pleft+2*(pwid+pspace) pbot pwid pheight])
ax = contourf(obs.fr,1:num_hours,rmse_change,-50:50,'EdgeColor','none');
shading flat
colormap(mycolors)
caxis([-10 10])
grid on
xlim(myxlim)
xlabel('Frequency [Hz]')
set(gca,'YTickLabel',[])


chan = colorbar('Location','NorthOutside','position',[pleft pbot+pheight+.01 1-pleft-pright .02]);
ylabel(chan,'Change \theta_m RMSE [deg]')
printFig(gcf,'fig_dir_rmse_fh',[6.5 4],'png')





