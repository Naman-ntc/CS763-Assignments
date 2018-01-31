function [entropy] = jointEntropy(movingImage, originalImage)

    [h, w] = size(movingImage);
    movingImage = floor(movingImage/10) + 1;
    originalImage = floor(originalImage/10) + 1;
    movingImage = reshape(movingImage, h*w, 1);
    originalImage = reshape(originalImage, h*w, 1);
    indices = [movingImage, originalImage];

    histo = zeros(26, 26);

    for i=1:size(indices, 1)
        histo(indices(i, 1), indices(i, 2)) = histo(indices(i, 1), indices(i, 2)) + 1; 
    end

    remove0s = 1 - (histo > 0);
    histo  = histo/(h*w);
    entropy = -sum(sum( histo.*log(histo + remove0s) ));


end