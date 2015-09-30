% Author: Davide Berdin

clear;

male_names = {'Jeremy', 'Lenny', 'Philip'};
female_names = {'Joyce','Marty','Niki'};
filename_tile = {'a_piece_of_cake','blow_a_fuse','catch_some_zs','down_to_the_wire','eager_beaver','fair_and_square', 'get_cold_feet', 'mellow_out','pulling_your_legs','thinking_out_loud'};

directory = '/Users/dado/Documents/University/Courses/Master-Thesis/Project/MATLAB/Smoothed_CSV_files/male/';
results = '/Users/dado/Documents/University/Courses/Master-Thesis/Project/MATLAB/small_multiples/male/';
Files = dir(strcat(directory,'*.csv'));

SUBPLOT_LINES = 4;
SUBPLOT_COLUMNS = 3;

image_counter = 1;
for tit = 1:10
    small_mult_fig = figure('Visible', 'on', 'units','normalized','Position',[0 0 1 1]);
    
    TITLE_SELECTOR = tit;
    
    intensity = [];
    f1 = [];
    f2 = [];
    f3 = [];

    for na = 1:3
        NAME_SELECTOR = na;

        for f = 1:length(Files)

            filename = Files(f).name;
            name = male_names(NAME_SELECTOR);
            title_filename = filename_tile(TITLE_SELECTOR);

            if (not(isempty(strfind(filename, name))) && not(isempty(strfind(filename, title_filename))))
                display(strcat('Operating on: ', filename))

               temp_filename = strcat(directory, filename);
               [time_, intensity_, f1_, f2_, f3_] = extract_features(temp_filename, false);

               time = time_;
               intensity = intensity_;
               f1 = f1_;
               f2 = f2_;
               f3 = f3_;
            else
                continue
            end
        end

        clear title xlabel ylabel;
        title_name = name{1};
        % start suplotting stuff
        % INTENSITY
        a = subplot_tight(SUBPLOT_COLUMNS, SUBPLOT_LINES, image_counter, [0.05 0.05]);    
        image_counter = image_counter + 1;            
        %plot([averaged_intensity smoothed_intensity]);
        plot(intensity, 'r');
        set( get(a,'YLabel'), 'String', 'Intensity' );
        title(title_name);
        hold on;
        axis tight

        % F1
        b = subplot_tight(SUBPLOT_COLUMNS, SUBPLOT_LINES, image_counter, [0.05 0.05]);
        image_counter = image_counter + 1;
        %plot([averaged_f1, smoothed_f1]);
        plot(f1, 'b');
        set( get(b,'YLabel'), 'String', 'F1' );
        title(title_name);
        hold on;
        axis tight

        % F2
        c = subplot_tight(SUBPLOT_COLUMNS, SUBPLOT_LINES, image_counter, [0.05 0.05]);
        image_counter = image_counter + 1;
%         plot([averaged_f2, smoothed_f2]);
        plot(f2, 'b');
        set( get(c,'YLabel'), 'String', 'F2' );
        title(title_name);
        hold on;
        axis tight

        % F3
        d = subplot_tight(SUBPLOT_COLUMNS, SUBPLOT_LINES, image_counter, [0.05 0.05]);
        image_counter = image_counter + 1;
%         plot([averaged_f3, smoothed_f3]);
        plot(f3, 'b');
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

