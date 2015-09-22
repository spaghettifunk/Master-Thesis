function [feat1, feat2, feat3, feat4, max_length] = set_equal_length(feat1, feat2, feat3, feat4)
    S1 = size(feat1);
    S2 = size(feat2);
    S3 = size(feat3);
    S4 = size(feat4);

    MaxS = max([S1, S2, S3, S4]);

    if MaxS > S1(1); feat1(end+1:MaxS(1), :) = 0; end
    if MaxS > S2(1); feat2(end+1:MaxS(1), :) = 0; end
    if MaxS > S3(1); feat3(end+1:MaxS(1), :) = 0; end
    if MaxS > S4(1); feat4(end+1:MaxS(1), :) = 0; end
    
    max_length = MaxS;
end