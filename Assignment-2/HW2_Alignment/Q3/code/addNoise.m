function [noisyImage] = addNoise(image)
    noisyImage = image + (randi(8, size(image)) - 1);
    noisyImage = min(noisyImage, 255);
    noisyImage = max(noisyImage, 0);
end