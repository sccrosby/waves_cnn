function [ obs, ww3, pred] = load_data( fol, fname, num_hours )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

obs = load(sprintf('%s/%s_obs_data.mat',fol,fname));
ww3 = load(sprintf('%s/%s_ww3_data.mat',fol,fname));
for hr = 1:num_hours
    pred(hr) = load(sprintf('%s/%s_hr%d_data.mat',fol,fname,hr));
end

% CHECK CIRCSHIFT, why 12-hours? 
% Corrections Applied (Appears that pred have a 12-hour offset, Check for BUG!!
obs.fr = [0.04 0.045 0.05 0.055 0.06 0.065 0.07 0.075 0.08 0.085 0.09 0.095 0.10125 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25];
ww3.fr = obs.fr;
% for hr = 1:num_hours
%     pred(hr).e = circshift(pred(hr).e,[12 0]);
%     pred(hr).a1 = circshift(pred(hr).a1,[12 0]);
%     pred(hr).b1 = circshift(pred(hr).b1,[12 0]);
%     pred(hr).a2 = circshift(pred(hr).a2,[12 0]);
%     pred(hr).b2 = circshift(pred(hr).b2,[12 0]);
%     pred(hr).fr = obs.fr;    
% end



end

