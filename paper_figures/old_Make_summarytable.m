
clearvars

% set data directory
data_dir = set_data_dir();


fol = strcat(data_dir, '46211_20190811014904_dev_test_6-24_2019_08_22_091752');
% fol = strcat(data_dir, '46214_20190809181416_dev_test_6-24_2019_08_22_085425');
% fol = strcat(data_dir, '46218_20190808160557_dev_test_6-24_2019_08_22_084556');


[hs_rmse_ww3_br, hs_rmse_pred_br, hs_rmse_pred, tm_rmse_ww3, tm_rmse_pred, ...
    dir_rmse_ww3, dir_rmse_pred, bias] = get_rmse_all( fol );

inds = {1:3,3:6,6:12,12:24};

for ii = 1:length(inds)
    dhs(ii) = 1-mean(hs_rmse_pred_br(inds{ii}))/hs_rmse_ww3_br;
    ddir(ii) = 1-mean(dir_rmse_pred(inds{ii}))/dir_rmse_ww3;
    dtm(ii) = 1-mean(tm_rmse_pred(inds{ii}))/tm_rmse_ww3;
end

fprintf('1-3, 3-6, 6-12, 12-24\n')
disp([dhs; ddir; dtm] * 100)
