
clearvars

fol = '46211_20190716000944_dev_dev_48-24_2019_07_16_094407';

[hs_rmse_ww3_br, hs_rmse_pred_br, tm_rmse_ww3, tm_rmse_pred, ...
    dir_rmse_ww3, dir_rmse_pred] = get_rmse_all( fol );

inds = {1:3,3:6,6:12,12:24};

for ii = 1:length(inds)
    dhs(ii) = 1-mean(hs_rmse_pred_br(inds{ii}))/hs_rmse_ww3_br;
    ddir(ii) = 1-mean(dir_rmse_pred(inds{ii}))/dir_rmse_ww3;
    dtm(ii) = 1-mean(tm_rmse_pred(inds{ii}))/tm_rmse_ww3;
end

fprintf('1-3, 3-6, 6-12, 12-24\n')
disp([dhs; ddir; dtm])
