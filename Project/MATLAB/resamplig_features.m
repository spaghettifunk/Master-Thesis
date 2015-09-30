function data_res = resamplig_features(time, data)
%         data_TS = timeseries(data,time); % define the data as a timeseries 
%         new_time = time(0):dt:time(end); % new vector of time with fixed dt=1/fs
%         data_res = resample(data_TS,new_time); % data resampled at constant fs

        x = 1:numel(data);
        xp = linspace(x(1), x(end), numel(time)); %// Specify 3 output points
        data_res = interp1(x, data, xp); 
        data_res = data_res(:);
        
        %s = size(data_res);