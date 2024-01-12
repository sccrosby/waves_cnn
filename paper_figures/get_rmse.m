function [hs_sw_rmse_ww3, hs_sw_rmse_ww3_br, hs_sw_rmse_pred, hs_ss_rmse_ww3, ...
    hs_ss_rmse_ww3_br, hs_ss_rmse_pred, rmse_pred_SW, rmse_ww3_SW, ...
    rmse_pred_SS, rmse_ww3_SS, tm_sw_rmse_pred, tm_ss_rmse_pred, ...
    tm_sw_rmse_ww3, tm_ss_rmse_ww3] = get_rmse( fol )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

num_hours = 24;

files=dir(fol);
fname = files(4).name(1:17);
disp(fname)

% Load Data
[ obs, ww3, pred] = load_data( fol, fname, num_hours );

% Integrate for direction for swell
abnd = 1:13; %0.04-0.105 Hz
obsSW = intMeanDir( obs, 1, abnd );
ww3SW = intMeanDir( ww3, 1, abnd );
predSW = intMeanDir( pred, num_hours, abnd );

% Estimate Hs for SW and SS
ww3.hsSW = 4*sqrt(ww3.e(:,abnd)*ww3.bw(abnd)');
obs.hsSW = 4*sqrt(obs.e(:,abnd)*obs.bw(abnd)');
for hr = 1:num_hours
    pred(hr).hsSW = 4*sqrt(pred(hr).e(:,abnd)*pred(hr).bw(abnd)');
end

% Integrate for direction for seas
abnd = 14:28; %0.105 - 0.25 Hz
obsSS = intMeanDir( obs, 1, abnd );
ww3SS = intMeanDir( ww3, 1, abnd );
predSS = intMeanDir( pred, num_hours, abnd );

% Estimate Hs for SW and SS
ww3.hsSS = 4*sqrt(ww3.e(:,abnd)*ww3.bw(abnd)');
obs.hsSS = 4*sqrt(obs.e(:,abnd)*obs.bw(abnd)');
for hr = 1:num_hours
    pred(hr).hsSS = 4*sqrt(pred(hr).e(:,abnd)*pred(hr).bw(abnd)');
end

% Swell
% Estimate e-weighted RMSE:
temp = ww3SW.md1-obsSW.md1;
% fix wrapping
temp(temp>180) = temp(temp>180)-360;
temp(temp<-180) = temp(temp<-180)+360;
rmse_ww3_SW = sqrt(nansum(obs.hsSW.^2.*temp.^2)/nansum(obs.hsSW.^2));
for fh = 1:num_hours
    temp = predSW(fh).md1-obsSW.md1;
    % fix wrapping
    temp(temp>180) = temp(temp>180)-360;
    temp(temp<-180) = temp(temp<-180)+360;
    rmse_pred_SW(fh)= sqrt(nansum(obs.hsSW.^2.*temp.^2)/nansum(obs.hsSW.^2));
end

% Seas
temp = ww3SS.md1-obsSS.md1;
temp(temp>180) = temp(temp>180)-360;
temp(temp<-180) = temp(temp<-180)+360;
rmse_ww3_SS = sqrt(nansum(obs.hsSS.^2.*temp.^2)/nansum(obs.hsSS.^2));
for fh = 1:num_hours
    temp = predSS(fh).md1-obsSS.md1;
    temp(temp>180) = temp(temp>180)-360;
    temp(temp<-180) = temp(temp<-180)+360;
    rmse_pred_SS(fh)= sqrt(nansum(obs.hsSS.^2.*temp.^2)/nansum(obs.hsSS.^2));
end

% HS RMSE
bias = mean(ww3.hsSS - obs.hsSS);
hs_ss_rmse_ww3 = sqrt(mean((ww3.hsSS - obs.hsSS).^2));
hs_ss_rmse_ww3_br = sqrt(mean((ww3.hsSS - obs.hsSS-bias).^2));
for fh = 1:num_hours
hs_ss_rmse_pred(fh) = sqrt(nanmean((pred(fh).hsSS - obs.hsSS).^2));
end

bias = mean(ww3.hsSW - obs.hsSW);
hs_sw_rmse_ww3 = sqrt(mean((ww3.hsSW - obs.hsSW).^2));
hs_sw_rmse_ww3_br = sqrt(mean((ww3.hsSW - obs.hsSW-bias).^2));
for fh = 1:num_hours
hs_sw_rmse_pred(fh) = sqrt(nanmean((pred(fh).hsSW - obs.hsSW).^2));
end

% Calc Tm,
abnd = 1:13; %0.04-0.105 Hz
obs.fm_sw = sum(obs.e(:,abnd).*repmat(obs.fr(:,abnd),[length(obs.time) 1]),2)./sum(obs.e(:,abnd),2);
obs.Tm_sw = 1./obs.fm_sw;
ww3.fm_sw = sum(ww3.e(:,abnd).*repmat(ww3.fr(:,abnd),[length(ww3.time) 1]),2)./sum(ww3.e(:,abnd),2);
ww3.Tm_sw = 1./ww3.fm_sw;
for fh = 1:num_hours
    pred(fh).fm_sw = sum(pred(fh).e(:,abnd).*repmat(pred(fh).fr(:,abnd),[length(pred(fh).time) 1]),2)./sum(pred(fh).e(:,abnd),2);
    pred(fh).Tm_sw = 1./pred(fh).fm_sw;
end
abnd = 14:28; %0.105 - 0.25 Hz
obs.fm_ss = sum(obs.e(:,abnd).*repmat(obs.fr(:,abnd),[length(obs.time) 1]),2)./sum(obs.e(:,abnd),2);
obs.Tm_ss = 1./obs.fm_ss;
ww3.fm_ss = sum(ww3.e(:,abnd).*repmat(ww3.fr(:,abnd),[length(ww3.time) 1]),2)./sum(ww3.e(:,abnd),2);
ww3.Tm_ss = 1./ww3.fm_ss;
for fh = 1:num_hours
    pred(fh).fm_ss = sum(pred(fh).e(:,abnd).*repmat(pred(fh).fr(:,abnd),[length(pred(fh).time) 1]),2)./sum(pred(fh).e(:,abnd),2);
    pred(fh).Tm_ss = 1./pred(fh).fm_ss;
end

% Tm and RMSE
tm_sw_rmse_ww3 = sqrt(nansum(obs.hsSW.^2.*(ww3.Tm_sw-obs.Tm_sw).^2)/nansum(obs.hsSW.^2));
for fh = 1:num_hours
    tm_sw_rmse_pred(fh) = sqrt(nansum(obs.hsSW.^2.*(pred(fh).Tm_sw-obs.Tm_sw).^2)/nansum(obs.hsSW.^2));
end
tm_ss_rmse_ww3 = sqrt(nansum(obs.hsSS.^2.*(ww3.Tm_ss-obs.Tm_ss).^2)/nansum(obs.hsSS.^2));
for fh = 1:num_hours
    tm_ss_rmse_pred(fh) = sqrt(nansum(obs.hsSS.^2.*(pred(fh).Tm_ss-obs.Tm_ss).^2)/nansum(obs.hsSS.^2));
end

end
