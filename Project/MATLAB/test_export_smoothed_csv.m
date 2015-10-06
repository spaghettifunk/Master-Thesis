% Author: Davide Berdin

clear;

filename_tile = {'a_piece_of_cake','blow_a_fuse','catch_some_zs','down_to_the_wire','eager_beaver','fair_and_square', 'get_cold_feet', 'mellow_out','pulling_your_legs','thinking_out_loud'};

% test
directory = '/Users/dado/Documents/University/Courses/Master-Thesis/Project/src/input-data/test-csv-files/';
results = '/Users/dado/Documents/University/Courses/Master-Thesis/Project/src/output-data/test-smoothed-csv-files/';
Files = dir(strcat(directory,'*.csv'));

SUBPLOT_LINES = 4;
SUBPLOT_COLUMNS = 3;

image_counter = 1;
for i = 1:11
    TITLE_SELECTOR = i;

    filename = Files(i).name;

    display(strcat('Operating on: ', filename))  
    if strcmp(filename, '.csv')
        continue
    end
    
   title_filename = filename_tile(TITLE_SELECTOR);

   temp_filename = strcat(directory, filename);
   [time, pitch, intensity, f1, f2, f3] = extract_features(temp_filename, true);

    smoothed_intensity = apply_filter(intensity, 'exponential');
    smoothed_f1 = apply_filter(f1, 'exponential');
    smoothed_f2 = apply_filter(f2, 'exponential');
    smoothed_f3 = apply_filter(f3, 'exponential');

    % resampling
    time = [.0:.01:1.6]; % is your starting vector of time
    time = time(:);
    smoothed_intensity = resamplig_features(time, smoothed_intensity);
    smoothed_f1 = resamplig_features(time, smoothed_f1);
    smoothed_f2 = resamplig_features(time, smoothed_f2);
    smoothed_f3 = resamplig_features(time, smoothed_f3);

    smoothedTable = table(time, smoothed_intensity, smoothed_f1, smoothed_f2, smoothed_f3);

    composed_filename = strcat(strcat('Davide', '_'), title_filename);
    filename = composed_filename;
    save_result = strcat(results, strcat(filename, '.csv'));
    saved_filename = save_result{1};
    writetable(smoothedTable, saved_filename);
end

