clearvars

fid = fopen('noaa_buoys.txt','r');
data = textscan(fid,'%s');
fclose(fid);

id = data{1};

for ii = 1:length(id)

M(ii) = getndbc( id{ii}, 1979 );

end

save('buoy_bulkwave_met_obs.mat','M','id')
