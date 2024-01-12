% clear vars
% 
% addpath cbrewer\
% 
% % Set bounds for time series plots (Later loop for a couple different months during winters)
% date_start = datenum(2004,12,1);
% date_end = datenum(2005,2,1); 
% 
% fol = 'C:\Users\Owner\Desktop\noah_research\GraysHarbor\b46211\46211_20211106172742_dev_dev_12-24_2021_11_17_205421';
% fname = '2021_11_17_205421';
% num_hours = 24;
% [ obs, ww3, pred] = load_data( fol, fname, num_hours );
% return
% Load step
%GraysDFol = ["46211_20211106172742_dev_dev_12-24_2022_01_13_172020", "46211_20211116153729_dev_dev_12-24_2022_01_13_183927", "46211_20211116202456_dev_dev_12-24_2022_01_15_213317", "46211_20211117210845_dev_dev_12-24_2022_01_15_213632", "46211_20211118155219_dev_dev_12-24_2022_01_15_213951", "46211_20211118181934_dev_dev_12-24_2022_01_15_214217", "46211_20211118212705_dev_dev_12-24_2022_01_15_214413", "46211_20211119215211_dev_dev_12-24_2022_01_15_214735", "46211_20211120132657_dev_dev_12-24_2022_01_15_214959"];
%GraysDPre = ["2022_01_13_172020", "2022_01_13_183927", "2022_01_15_213317", "2022_01_15_213632", "2022_01_15_213951", "2022_01_15_214217", "2022_01_15_214413", "2022_01_15_214735", "2022_01_15_214959"];
%GraysTFol = ["46211_20211106172742_dev_test_12-24_2022_01_13_172051", "46211_20211116153729_dev_test_12-24_2022_01_13_183957", "46211_20211116202456_dev_test_12-24_2022_01_15_213348", "46211_20211117210845_dev_test_12-24_2022_01_15_213702", "46211_20211118155219_dev_test_12-24_2022_01_15_214022", "46211_20211118181934_dev_test_12-24_2022_01_15_214248", "46211_20211118212705_dev_test_12-24_2022_01_15_214444", "46211_20211119215211_dev_test_12-24_2022_01_15_214806", "46211_20211120132657_dev_test_12-24_2022_01_15_215032"];
%GraysTPre = ["2022_01_13_172051", "2022_01_13_183957", "2022_01_15_213348", "2022_01_15_213702", "2022_01_15_214022", "2022_01_15_214248", "2022_01_15_214444", "2022_01_15_214806", "2022_01_15_215032"];
%Grays = [GraysDFol; GraysDPre];

%ReyesDFol = ["46214_20211130132403_dev_dev_12-24_2022_01_18_183802", "46214_20211130150622_dev_dev_12-24_2022_01_18_184016", "46214_20211130170045_dev_dev_12-24_2022_01_18_184305", "46214_20211130205429_dev_dev_12-24_2022_01_18_184512", "46214_20211201210325_dev_dev_12-24_2022_01_18_184703", "46214_20211203220013_dev_dev_12-24_2022_01_18_184852", "46214_20220110230947_dev_dev_12-24_2022_01_18_185055", "46214_20220111143056_dev_dev_12-24_2022_01_18_185317", "46214_20220111154223_dev_dev_12-24_2022_01_18_185858"];
%ReyesDPre = ["2022_01_18_183802", "2022_01_18_184016", "2022_01_18_184305", "2022_01_18_184512","2022_01_18_184703", "2022_01_18_184852", "2022_01_18_185055", "2022_01_18_185317", "2022_01_18_185858"];
%ReyesTFol = ["46214_20211130132403_dev_test_12-24_2022_01_18_183830", "46214_20211130150622_dev_test_12-24_2022_01_18_184045", "46214_20211130170045_dev_test_12-24_2022_01_18_184333", "46214_20211130205429_dev_test_12-24_2022_01_18_184541", "46214_20211201210325_dev_test_12-24_2022_01_18_184732", "46214_20211203220013_dev_test_12-24_2022_01_18_184921", "46214_20220110230947_dev_test_12-24_2022_01_18_185124", "46214_20220111143056_dev_test_12-24_2022_01_18_185346", "46214_20220111154223_dev_test_12-24_2022_01_18_185926"];
%ReyesTPre = ["2022_01_18_183830", "2022_01_18_184045", "2022_01_18_184333", "2022_01_18_184541", "2022_01_18_184732", "2022_01_18_184921", "2022_01_18_185124", "2022_01_18_185346", "2022_01_18_185926"];
%Reyes = [ReyesTFol; ReyesTPre];

HarvestDFol = ["46218_20220111180456_dev_dev_12-24_2022_01_19_130424", "46218_20220111235037_dev_dev_12-24_2022_01_19_130707", "46218_20220112153721_dev_dev_12-24_2022_01_19_130957", "46218_20220112171646_dev_dev_12-24_2022_01_19_131355", "46218_20220112181040_dev_dev_12-24_2022_01_19_131628", "46218_20220112194345_dev_dev_12-24_2022_01_19_131831", "46218_20220112222318_dev_dev_12-24_2022_01_19_132026", "46218_20220113101720_dev_dev_12-24_2022_01_19_132309", "46218_20220113115345_dev_dev_12-24_2022_01_19_132455", "46218_20220113142037_dev_dev_12-24_2022_01_19_132730"];
HarvestDPre = ["2022_01_19_130424", "2022_01_19_130707", "2022_01_19_130957", "2022_01_19_131355", "2022_01_19_131628", "2022_01_19_131831", "2022_01_19_132026", "2022_01_19_132309", "2022_01_19_132455", "2022_01_19_132730"];
HarvestTFol = ["46218_20220111180456_dev_test_12-24_2022_01_19_130452", "46218_20220111235037_dev_test_12-24_2022_01_19_130734", "46218_20220112153721_dev_test_12-24_2022_01_19_131024", "46218_20220112171646_dev_test_12-24_2022_01_19_131423", "46218_20220112181040_dev_test_12-24_2022_01_19_131655", "46218_20220112194345_dev_test_12-24_2022_01_19_131858", "46218_20220112222318_dev_test_12-24_2022_01_19_132053", "46218_20220113101720_dev_test_12-24_2022_01_19_132336", "46218_20220113115345_dev_test_12-24_2022_01_19_132523", "46218_20220113142037_dev_test_12-24_2022_01_19_132757"];
HarvestTPre = ["2022_01_19_130452", "2022_01_19_130734", "2022_01_19_131024", "2022_01_19_131423", "2022_01_19_131655", "2022_01_19_131858", "2022_01_19_132053", "2022_01_19_132336", "2022_01_19_132523", "2022_01_19_132757"];
Harvest = [HarvestDFol; HarvestDPre];
% Specify folder for correct hours used in train
%buoy = "GraysHarbor";
%bnum = "b46211";
%buoy = "PointReyes";
%bnum = "b46214";
buoy = "Harvest";
bnum = "b46218";
for yr_train = 1:10
    disp(yr_train);
    fol = Harvest(1,yr_train);%sprintf('%s_%d',fol_train, yr_train);
    prefix = Harvest(2, yr_train); % ? vary
    num_hours = 24;
    [ A(yr_train).obs, A(yr_train).ww3, A(yr_train).pred] = load_data(buoy, bnum, fol, prefix, num_hours );

% Go and change everything to A(yr_train).ETC
% Integrate for Hs (Significant Wave Height)
A(yr_train).ww3.hs = 4*sqrt(A(yr_train).ww3.e*A(yr_train).ww3.bw');
A(yr_train).obs.hs = 4*sqrt(A(yr_train).obs.e*A(yr_train).obs.bw');
for hr = 1:num_hours
    A(yr_train).pred(hr).hs = 4*sqrt(A(yr_train).pred(hr).e*A(yr_train).pred(hr).bw');
end

% Calc Tm (mean wave period)
%   First calculate mean freq, this is a weight average at each time step
%   of fr and e. Then Tm = 1/fm
A(yr_train).ww3.fm = sum(A(yr_train).ww3.e.*repmat(A(yr_train).ww3.fr,[length(A(yr_train).ww3.time) 1]),2)./sum(A(yr_train).ww3.e,2);
A(yr_train).ww3.Tm = 1./A(yr_train).ww3.fm;
A(yr_train).obs.fm = sum(A(yr_train).obs.e.*repmat(A(yr_train).obs.fr,[length(A(yr_train).obs.time) 1]),2)./sum(A(yr_train).obs.e,2);
A(yr_train).obs.Tm = 1./A(yr_train).obs.fm;
for hr = 1:num_hours
    A(yr_train).pred(hr).fm = sum(A(yr_train).pred(hr).e.*repmat(A(yr_train).pred(hr).fr,[length(A(yr_train).pred(hr).time) 1]),2)./sum(A(yr_train).pred(hr).e,2);
    A(yr_train).pred(hr).Tm = 1./A(yr_train).pred(hr).fm;
end

% Estimate directional integrated across frequency
[A(yr_train).obs.md1A,A(yr_train).obs.md2A,A(yr_train).obs.spr1A,A(yr_train).obs.spr2A,~,~]=getkuikstats(sum(A(yr_train).obs.a1,2)./sum(A(yr_train).obs.e,2),sum(A(yr_train).obs.b1,2)./sum(A(yr_train).obs.e,2),sum(A(yr_train).obs.a2,2)./sum(A(yr_train).obs.e,2),sum(A(yr_train).obs.b2,2)./sum(A(yr_train).obs.e,2));
[A(yr_train).ww3.md1A,A(yr_train).ww3.md2A,A(yr_train).ww3.spr1A,A(yr_train).ww3.spr2A,~,~]=getkuikstats(sum(A(yr_train).ww3.a1,2)./sum(A(yr_train).ww3.e,2),sum(A(yr_train).ww3.b1,2)./sum(A(yr_train).ww3.e,2),sum(A(yr_train).ww3.a2,2)./sum(A(yr_train).ww3.e,2),sum(A(yr_train).ww3.b2,2)./sum(A(yr_train).ww3.e,2));
for hr = 1:num_hours
    [A(yr_train).pred(hr).md1A,A(yr_train).pred(hr).md2A,A(yr_train).pred(hr).spr1A,A(yr_train).pred(hr).spr2A,~,~]=getkuikstats(sum(A(yr_train).pred(hr).a1,2)./sum(A(yr_train).pred(hr).e,2),sum(A(yr_train).pred(hr).b1,2)./sum(A(yr_train).pred(hr).e,2),sum(A(yr_train).pred(hr).a2,2)./sum(A(yr_train).pred(hr).e,2),sum(A(yr_train).pred(hr).b2,2)./sum(A(yr_train).pred(hr).e,2));
end


%----------------------------------------------------------------------------------------------
% CALC RMSE
% Loop over data used in train
    % Loop over forecast hour
for hr_for = 1:num_hours
    rmse(yr_train,hr_for) = sqrt(nanmean( (A(yr_train).pred(hr_for).md1A - A(yr_train).obs.md1A).^2));
    rmseww3(yr_train,hr_for) = sqrt(nanmean( (A(yr_train).ww3.md1A - A(yr_train).obs.md1A).^2));
    absdiff = rmse - rmseww3; %absolute difference: plotting the (rmseww3 - rmse)
    reldiff = ((rmseww3 - rmse)./rmseww3);%relative difference: plotting the (rmseww3 - rmse) / rmseww3
end
%----------------------------------------------------------------------------------------------------
end

%% Plot heatmap
clf
[X, Y] = meshgrid(1:10,1:24);
contourf(X,Y,absdiff',20,'EdgeColor','none');
shading flat
chan = colorbar('Location','NorthOutside');
colormap("jet");
%caxis([-.5 .5])
ylabel('Forecast time [Hr]')
xlabel('Length of Training Set (Years)')
ylabel(chan,'RMSE (Meters)')
printFig(gcf,'RMSE_Loss_relative',[8 8],'pdf')