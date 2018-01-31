%% MyMainScript

tic;
%% Your code here

%{
movingImage1 = rotateAndTranslate(imread('../input/negative_barbara.png'), 23.5, -3);
movingImage2 = rotateAndTranslate(imread('../input/noflash1.jpg'), 23.5, -3);
save('Q3Data', movingImage1, movingImage2);
movingImage1 = addNoise(movingImage1);
movingImage2 = addNoise(movingImage2);
originalImage1 = imread('../input/barbara.png');
originalImage2 = imread('../input/flash1.jpg');
save('Q3Data', 'movingImage1', 'movingImage2', 'originalImage1', 'originalImage2');
%}
% Running rotation is slow because it uses interp

load('Q3Data');


%%downsample for now
movingImage = movingImage1(1:4:end, 1:4:end);
originalImage = originalImage1(1:4:end, 1:4:end);
entropyGrid1 = zeros(121, 24);

for i=33:42
    for j=13:17
        currentImage = rotateAndTranslate(movingImage, i - 61, j - 13);
        entropyGrid1(i, j) = jointEntropy(currentImage, originalImage);
        i, j
    end
end

toc;
