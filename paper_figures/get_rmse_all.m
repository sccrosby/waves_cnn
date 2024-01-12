function [hs_rmse_ww3_br, hs_rmse_pred_br, hs_rmse_pred, tm_rmse_ww3, tm_rmse_pred, ...
    dir_rmse_ww3, dir_rmse_pred, bias] = get_rmse_all( fol )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

bnd = 1:28; % Frequencies

num_hours = 24;

files=dir(fol);
fname = files(4).name(1:17);
disp(fname)

% Load Data
[ obs, ww3, pred] = load_data( fol, fname, num_hours );

% Integrate for direction
dir_obs = intMeanDir( obs, 1, bnd );
dir_ww3 = intMeanDir( ww3, 1, bnd );
dir_pred = intMeanDir( pred, num_hours, bnd );

% Estimate Hs for 
ww3.hs = 4*sqrt(ww3.e(:,bnd)*ww3.bw(bnd)');
obs.hs = 4*sqrt(obs.e(:,bnd)*obs.bw(bnd)');
for hr = 1:num_hours
    pred(hr).hs = 4*sqrt(pred(hr).e(:,bnd)*pred(hr).bw(bnd)');
end

% Estimate e-weighted RMSE for direction
temp = dir_ww3.md1-dir_obs.md1;
% fix wrapping
temp(temp>180) = temp(temp>180)-360;
temp(temp<-180) = temp(temp<-180)+360;
dir_rmse_ww3 = sqrt(nansum(obs.hs.^2.*temp.^2)/nansum(obs.hs.^2));
for fh = 1:num_hours
    temp = dir_pred(fh).md1-dir_obs.md1;
    % fix wrapping
    temp(temp>180) = temp(temp>180)-360;
    temp(temp<-180) = temp(temp<-180)+360;
    dir_rmse_pred(fh)= sqrt(nansum(obs.hs.^2.*temp.^2)/nansum(obs.hs.^2));
end

% HS RMSE
bias = mean(ww3.hs - obs.hs);
hs_rmse_ww3 = sqrt(mean((ww3.hs - obs.hs).^2));
hs_rmse_ww3_br = sqrt(mean((ww3.hs - obs.hs-bias).^2));
for fh = 1:num_hours
    bias(fh) = nanmean(pred(fh).hs - obs.hs);
    hs_rmse_pred(fh) = sqrt(nanmean((pred(fh).hs - obs.hs).^2));
    hs_rmse_pred_br(fh) = sqrt(nanmean((pred(fh).hs - obs.hs-bias(fh)).^2));
end

% Calc Tm,
obs.fm = sum(obs.e(:,bnd).*repmat(obs.fr(:,bnd),[length(obs.time) 1]),2)./sum(obs.e(:,bnd),2);
obs.Tm = 1./obs.fm;
ww3.fm = sum(ww3.e(:,bnd).*repmat(ww3.fr(:,bnd),[length(ww3.time) 1]),2)./sum(ww3.e(:,bnd),2);
ww3.Tm = 1./ww3.fm;
for fh = 1:num_hours
    pred(fh).fm = sum(pred(fh).e(:,bnd).*repmat(pred(fh).fr(:,bnd),[length(pred(fh).time) 1]),2)./sum(pred(fh).e(:,bnd),2);
    pred(fh).Tm = 1./pred(fh).fm;
end

% Tm and RMSE
tm_rmse_ww3 = sqrt(nansum(obs.hs.^2.*(ww3.Tm-obs.Tm).^2)/nansum(obs.hs.^2));
for fh = 1:num_hours
    tm_rmse_pred(fh) = sqrt(nansum(obs.hs.^2.*(pred(fh).Tm-obs.Tm).^2)/nansum(obs.hs.^2));
end

end

