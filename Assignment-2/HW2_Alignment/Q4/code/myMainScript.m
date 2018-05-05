% Load images.
Imgdata = imageDatastore('../input/pier/');
% Display images to be stitched
montage(Imgdata.Files)

% Read the first image from the image set.
I = readimage(Imgdata, 1);
numImages = numel(Imgdata.Files);
tforms(numImages) = projective2d(eye(3));

% Iterate over remaining image pairs
for n = 2:numImages
    I1 = readimage(Imgdata, n-1);
    I2 = readimage(Imgdata, n);
    [matched_pts1, matched_pts2] = get_matchedPoints(I1, I2);
    A = ransacHomography(matched_pts2, matched_pts1, 2);
    %iA = inv(A);
    tforms(n) = projective2d(A');
    tforms(n).T = tforms(n).T * tforms(n-1).T;
end

imageSize = size(I);  % all the images are the same size
tforms = recenter_transformer(tforms, imageSize);

% to find the minimum and maximum output limits
for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
end
xMin = min([1; xlim(:)]);
xMax = max([imageSize(2); xlim(:)]);
yMin = min([1; ylim(:)]);
yMax = max([imageSize(1); ylim(:)]);
panorama = get_panorama(xMin, xMax, yMin, yMax, tforms, Imgdata);

figure
imshow(panorama)