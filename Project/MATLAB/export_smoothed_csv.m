% Author: Davide Berdin

clear;

male_names = {'Jeremy', 'Lenny', 'Philip'};
female_names = {'Joyce','Marty','Niki'};
filename_tile = {'a_piece_of_cake','blow_a_fuse','catch_some_zs','down_to_the_wire','eager_beaver','fair_and_square', 'get_cold_feet', 'mellow_out','pulling_your_legs','thinking_out_loud'};

directory = '/Users/dado/Documents/University/Courses/Master-Thesis/Project/DNN/CSV_Files/female/';
results = '/Users/dado/Documents/University/Courses/Master-Thesis/Project/MATLAB/Smoothed_CSV_files/female/';
Files = dir(strcat(directory,'*.csv'));

SUBPLOT_LINES = 4;
SUBPLOT_COLUMNS = 3;

image_counter = 1;
for tit = 1:10
    TITLE_SELECTOR = tit;
    for na = 1:3
        NAME_SELECTOR = na;

        k = 1;
        
        time1 = [];
        time2 = [];
        time3 = [];
        time4 = [];

        intensity1 = [];
        intensity2 = [];
        intensity3 = [];
        intensity4 = [];

        f1_1 = [];
        f1_2 = [];
        f1_3 = [];
        f1_4 = [];

        f2_1 = [];
        f2_2 = [];
        f2_3 = [];
        f2_4 = [];

        f3_1 = [];
        f3_2 = [];
        f3_3 = [];
        f3_4 = [];
        
        for f = 1:length(Files)

            filename = Files(f).name;
            name = female_names(NAME_SELECTOR);
            title_filename = filename_tile(TITLE_SELECTOR);

            if (not(isempty(strfind(filename, name))) && not(isempty(strfind(filename, title_filename))))
                display(strcat('Operating on: ', filename))                            

               temp_filename = strcat(directory, filename);
               [time, pitch, intensity, f1, f2, f3] = extract_features(temp_filename, true);

               if k == 1
                   time1 = time;
                   intensity1 = intensity;
                   f1_1 = f1;
                   f2_1 = f2;
                   f3_1 = f3;
               elseif k == 2
                   time2 = time;
                   intensity2 = intensity;
                   f1_2 = f1;
                   f2_2 = f2;
                   f3_2 = f3;
               elseif k == 3
                   time3 = time;
                   intensity3 = intensity;
                   f1_3 = f1;
                   f2_3 = f2;
                   f3_3 = f3;
               elseif k == 4
                   time4 = time;
                   intensity4 = intensity;
                   f1_4 = f1;
                   f2_4 = f2;
                   f3_4 = f3;       
               end

               k = k+1;
            else
                continue
            end
        end

        max_length = 0;
        [intensity1, intensity2, intensity3, intensity4, max_length] = set_equal_length(intensity1, intensity2, intensity3, intensity4);
        [f1_1, f1_2, f1_3, f1_4, max_length] = set_equal_length(f1_1, f1_2, f1_3, f1_4);
        [f2_1, f2_2, f2_3, f2_4, max_length] = set_equal_length(f2_1, f2_2, f2_3, f2_4);
        [f3_1, f3_2, f3_3, f3_4, max_length] = set_equal_length(f3_1, f3_2, f3_3, f3_4);

        averaged_intensity = [];
        averaged_f1 = [];
        averaged_f2 = [];
        averaged_f3 = [];
        for i=max_length
            avg_int = (intensity1(1:i) + intensity2(1:i) + intensity3(1:i) + intensity4(1:i)) / 4;
            avg_f1 = (f1_1(1:i) + f1_2(1:i) + f1_3(1:i) + f1_4(1:i)) / 4;
            avg_f2 = (f2_1(1:i) + f2_2(1:i) + f2_3(1:i) + f2_4(1:i)) / 4;
            avg_f3 = (f3_1(1:i) + f3_2(1:i) + f3_3(1:i) + f3_4(1:i)) / 4;

            averaged_intensity = [averaged_intensity avg_int];
            averaged_f1 = [averaged_f1 avg_f1];
            averaged_f2 = [averaged_f2 avg_f2];
            averaged_f3 = [averaged_f3 avg_f3];
        end

        % remove 0s on the beginning and on the end
        averaged_intensity = averaged_intensity(find(averaged_intensity,1,'first'):find(averaged_intensity,1,'last'));
        averaged_f1 = averaged_f1(find(averaged_f1,1,'first'):find(averaged_f1,1,'last'));
        averaged_f2 = averaged_f2(find(averaged_f2,1,'first'):find(averaged_f2,1,'last'));
        averaged_f3 = averaged_f3(find(averaged_f3,1,'first'):find(averaged_f3,1,'last'));

        smoothed_intensity = apply_filter(averaged_intensity, 'exponential');
        smoothed_f1 = apply_filter(averaged_f1, 'exponential');
        smoothed_f2 = apply_filter(averaged_f2, 'exponential');
        smoothed_f3 = apply_filter(averaged_f3, 'exponential');
               
        % resampling
        time = [.0:.01:1.6]; % is your starting vector of time
        time = time(:);
        smoothed_intensity = resamplig_features(time, smoothed_intensity);
        smoothed_f1 = resamplig_features(time, smoothed_f1);
        smoothed_f2 = resamplig_features(time, smoothed_f2);
        smoothed_f3 = resamplig_features(time, smoothed_f3);
        
        smoothedTable = table(time, smoothed_intensity, smoothed_f1, smoothed_f2, smoothed_f3);
        
        composed_filename = strcat(strcat(name, '_'), title_filename);
        filename = composed_filename;
        save_result = strcat(results, strcat(filename, '.csv'));
        saved_filename = save_result{1};
        writetable(smoothedTable, saved_filename);
    end
end

