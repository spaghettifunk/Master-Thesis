function [feat1, feat2, feat3, feat4, feat5, max_length] = set_equal_length2(feat1, feat2, feat3, feat4, feat5)
    S1 = size(feat1);
    S2 = size(feat2);
    S3 = size(feat3);
    S4 = size(feat4);
    S5 = size(feat5);

    MaxS = max([S1, S2, S3, S4, S5]);

    if MaxS > S1(1); feat1(end+1:MaxS(1), :) = 0; end
    if MaxS > S2(1); feat2(end+1:MaxS(1), :) = 0; end
    if MaxS > S3(1); feat3(end+1:MaxS(1), :) = 0; end
    if MaxS > S4(1); feat4(end+1:MaxS(1), :) = 0; end
    if MaxS > S5(1); feat5(end+1:MaxS(1), :) = 0; end
    
    max_length = MaxS;
end