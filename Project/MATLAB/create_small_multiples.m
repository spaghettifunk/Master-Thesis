% Author: Davide Berdin

clear;

male_names = {'Jeremy', 'Lenny', 'Philip'};
female_names = {'Joyce','Marty','Niki'};
filename_tile = {'a_piece_of_cake','blow_a_fuse','catch_some_zs','down_to_the_wire','eager_beaver','fair_and_square', 'get_cold_feet', 'mellow_out','pulling_your_legs','thinking_out_loud'};

directory = '/Users/dado/Documents/University/Courses/Master-Thesis/Project/DNN/CSV_Files/male/';
results = '/Users/dado/Documents/University/Courses/Master-Thesis/Project/MATLAB/small_multiples/male/';
Files = dir(strcat(directory,'*.csv'));

SUBPLOT_LINES = 4;
SUBPLOT_COLUMNS = 3;

image_counter = 1;
for tit = 1:10
    small_mult_fig = figure('Visible', 'on', 'units','normalized','Position',[0 0 1 1]);
    
    TITLE_SELECTOR = tit;
    
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
    for na = 1:3
        NAME_SELECTOR = na;

        k = 1;
        for f = 1:length(Files)

            filename = Files(f).name;
            name = male_names(NAME_SELECTOR);
            title_filename = filename_tile(TITLE_SELECTOR);

            if (not(isempty(strfind(filename, name))) && not(isempty(strfind(filename, title_filename))))
                display(strcat('Operating on: ', filename))

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

        smoothed_intensity = apply_filter(averaged_intensity, 'exponential');
        smoothed_f1 = apply_filter(averaged_f1, 'exponential');
        smoothed_f2 = apply_filter(averaged_f2, 'exponential');
        smoothed_f3 = apply_filter(averaged_f3, 'exponential');

        clear title xlabel ylabel;
        title_name = name{1};
        % start suplotting stuff
        % INTENSITY
        a = subplot_tight(SUBPLOT_COLUMNS, SUBPLOT_LINES, image_counter, [0.05 0.05]);    
        image_counter = image_counter + 1;            
        plot([averaged_intensity smoothed_intensity]);
        set( get(a,'YLabel'), 'String', 'Intensity' );
        title(title_name);
        hold on;
        axis tight

        % F1
        b = subplot_tight(SUBPLOT_COLUMNS, SUBPLOT_LINES, image_counter, [0.05 0.05]);
        image_counter = image_counter + 1;
        plot([averaged_f1, smoothed_f1]);
        set( get(b,'YLabel'), 'String', 'F1' );
        title(title_name);
        hold on;
        axis tight

        % F2
        c = subplot_tight(SUBPLOT_COLUMNS, SUBPLOT_LINES, image_counter, [0.05 0.05]);
        image_counter = image_counter + 1;
        plot([averaged_f2, smoothed_f2]);
        set( get(c,'YLabel'), 'String', 'F2' );
        title(title_name);
        hold on;
        axis tight

        % F3
        d = subplot_tight(SUBPLOT_COLUMNS, SUBPLOT_LINES, image_counter, [0.05 0.05]);
        image_counter = image_counter + 1;
        plot([averaged_f3, smoothed_f3]);
        set(get(d,'YLabel'), 'String', 'F3' );
        title(title_name);
        hold on;
        axis tight

        % DEBUG
        display(strcat('Saving Figure: ', filename))
    end
    
    % create graph and save it
    ha = axes('Position',[0 0 1 1],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');

    composed_filename = strcat(strcat(name, '_'), title_filename);
    title_fig = strrep(composed_filename, '_', ' ');
    text(0.5, 1, title_fig, 'HorizontalAlignment','center','VerticalAlignment', 'top');

    filename = composed_filename;
    save_result = strcat(results, strcat(filename, '.eps'));
    saved_filename = save_result{1};
    
    %set(gcf, 'Units','normalized','Position',[0 0 1 1]); %fullscreen
    %saveas(gcf, saved_filename, 'epsc2');
    %close all;
end

