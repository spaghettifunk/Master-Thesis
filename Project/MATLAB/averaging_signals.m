% Author: Davide Berdin
% Script for smoothing the signals given a csv file as input

clear;

male_names = {'Jeremy', 'Lenny', 'Philip'};
female_names = {'Joyce','Marty','Niki'};
filename_tile = {'a_piece_of_cake','blow_a_fuse','catch_some_zs','down_to_the_wire','eager_beaver','fair_and_square', 'get_cold_feet', 'mellow_out','pulling_your_legs','thinking_out_loud'};

directory = '/Users/dado/Documents/University/Courses/Master-Thesis/Project/DNN/CSV_Files/female/';
results = '/Users/dado/Documents/University/Courses/Master-Thesis/Project/MATLAB/female/';
Files = dir(strcat(directory,'*.csv'));

for na = 1:1
    NAME_SELECTOR = na;
    for tit = 1:10
        TITLE_SELECTOR = tit;

        k = 1;
        for f = 1:length(Files)

            filename = Files(f).name;
            name = female_names(NAME_SELECTOR);
            title = filename_tile(TITLE_SELECTOR);

            if (not(isempty(strfind(filename, name))) && not(isempty(strfind(filename, title))))
                display(strcat('Operating on: ', filename))

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

               temp_filename = strcat(directory, filename);
               [time, pitch, intensity, f1, f2, f3] = extract_features(temp_filename);

               if k == 1
                   %time1 = time;
                   intensity1 = intensity;
                   f1_1 = f1;
                   f2_1 = f2;
                   f3_1 = f3;
               elseif k == 2
                   %time2 = time;
                   intensity2 = intensity;
                   f1_2 = f1;
                   f2_2 = f2;
                   f3_2 = f3;
               elseif k == 3
                   %time3 = time;
                   intensity3 = intensity;
                   f1_3 = f1;
                   f2_3 = f2;
                   f3_3 = f3;
               elseif k == 4
                   %time4 = time;
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

        fig = figure('Visible','off');
        a = subplot(4,1,1);
        plot(averaged_intensity);
        %title(a, 'Intensity');
        axis tight

        b = subplot(4,1,2);
        plot(averaged_f1);
        %title(b, 'F1');
        axis tight

        c = subplot(4,1,3);
        plot(averaged_f2);
        %title(c, 'F2');
        axis tight

        d = subplot(4,1,4);
        plot(averaged_f3);
        %title(d, 'F3');
        axis tight

        ha = axes('Position',[0 0 1 1],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');

        composed_filename = strcat(strcat(name, '_'), title);
        title_fig = strrep(composed_filename, '_', ' ');
        text(0.5, 1, title_fig, 'HorizontalAlignment','center','VerticalAlignment', 'top');

        filename = composed_filename;
        save_result = strcat(results, filename);
        saveas(gcf, save_result{1}, 'png');

        display(strcat('Saving Figure: ', filename))
    end
end