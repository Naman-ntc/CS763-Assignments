function [new_tforms] = recenter_transformer(tforms, imageSize)
%RECENTER_TRANSFORMER Summary of this function goes here
%   Detailed explanation goes here
    for i = 1:numel(tforms)
        [xl(i,:), yl(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
    end
    meanXLim = mean(xl, 2);
    [~, idx] = sort(meanXLim);
    centerIdx = floor((numel(tforms)+1)/2);
    centerImageIdx = idx(centerIdx);
    Tinv = invert(tforms(centerImageIdx));
    for i = 1:numel(tforms)
        tforms(i).T = tforms(i).T * Tinv.T;
    end
    new_tforms = tforms;
end

