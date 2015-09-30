function [time, pitch, intensity, f1, f2, f3] = extract_features(filename, isExport)
    % read from csv
    fid = fopen(filename);
    
    if(isExport == false)
        headers = textscan(fid,'%s %s %s %s %s',1, 'delimiter',',');    % intensity, f1, f2, f3 -> add one moore for pitch
        data = textscan(fid,'%s %s %s %s %s','delimiter',',');          % intensity, f1, f2, f3 -> add one moore for pitch
    else
        headers = textscan(fid,'%s %s %s %s %s %s',1, 'delimiter',',');    % intensity, f1, f2, f3 -> add one moore for pitch
        data = textscan(fid,'%s %s %s %s %s %s','delimiter',',');          % intensity, f1, f2, f3 -> add one moore for pitch
    end
    
    fclose(fid);
    outCell = cell(size(data{1},1), length(headers));
    for i = 1:length(headers)
        if isnumeric(data{i})
            outCell(:,i) = num2cell(data{i});
        else
            outCell(:,i) = data{i};
        end
    end

    % convert
    structArray = cell2struct(outCell, [headers{:}], 2);
    table = struct2table(structArray);

    fields = fieldnames(table);
    % numOfRows = height(table);

    % get data
    t = table.(fields{1});
    p = table.(fields{2});
    i = table.(fields{3});
    f_1 = table.(fields{4});
    f_2 = table.(fields{5});
    f_3 = table.(fields{6});
    
    % struct for plotting
    time = str2double(t);
    pitch = str2double(p);
    intensity = str2double(i);
    f1 = str2double(f_1);
    f2 = str2double(f_2);
    f3 = str2double(f_3);
end