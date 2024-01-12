function [data_dir] = set_data_dir()

% built-in for jonny to run this on a linux machine
if isunix 
    if strfind(pwd, '/home/jonny/')  % Jonny's home computer
        data_dir = '/home/jonny/PycharmProjects/ml_waves19_jonny/data_outputs/';
        addpath /home/jonny/PycharmProjects/cbrewer/cbrewer;
    elseif strfind(pwd, '/home/mooneyj3')  % Jonny's school computer
        data_dir = '/home/hutch_research/projects/ml_waves19_jonny/data_outputs/';
        addpath /home/hutch_research/projects/ml_waves19/MatlabPlottingCode/cbrewer;
    else
        data_dir = '';
    end
end