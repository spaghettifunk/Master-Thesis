% for the Savitzky, we use the cubic filter
% for exponential, alpha is 0.45
function [value] = apply_filter(y, filtername)
    if strcmp(filtername, 'savitzky') == 1
        % Savitzky-Golay Filters
        value   = sgolayfilt(y, 3, 7);
    elseif strcmp(filtername, 'binomial') == 1
        % Binomial weighted Average
        h = [1/2 1/2];
        binomialCoeff = conv(h,h);
        for n = 1:4
            binomialCoeff = conv(binomialCoeff,h);
        end
        value = filter(binomialCoeff, 1, y);
    elseif strcmp(filtername, 'exponential') == 1
        % Exponential weighthed average
        alpha = 0.45;
        value = filter(alpha, [1 alpha-1], y);
    end
end