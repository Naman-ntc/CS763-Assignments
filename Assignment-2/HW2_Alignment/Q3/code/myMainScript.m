%% MyMainScript

tic;
%% Your code here

%{
movingImage1 = rotateAndTranslate('../input/negative_barbara.png');
movingImage2 = rotateAndTranslate('../input/noflash1.jpg');
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

toc;
