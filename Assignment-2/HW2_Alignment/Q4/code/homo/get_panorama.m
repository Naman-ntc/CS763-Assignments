function [panorama] = get_panorama(xMin, xMax, yMin, yMax, tforms, Imgdata)
%GET_PANORAMA Summary of this function goes here
%   Detailed explanation goes here
% reference used was matlab's offcial documentation/tutorial on stiching
    I = readimage(Imgdata, 1);
    numImages = numel(Imgdata.Files);
    pano_width  = round(xMax - xMin);
    pano_height = round(yMax - yMin);
    panorama = zeros([pano_height pano_width 3], 'like', I);
    blender = vision.AlphaBlender('Operation', 'Binary mask','MaskSource', 'Input port');
    xLimits = [xMin xMax];
    yLimits = [yMin yMax];
    panoramaView = imref2d([pano_height pano_width], xLimits, yLimits);

    for i = 1:numImages
        I = readimage(Imgdata, i);
        warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
        mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
        panorama = step(blender, panorama, warpedImage, mask);
    end
end

