entropyGrid1 = zeros(360, 512);
movingImage1 = double(imread('../input/negative_barbara.png'));
movingImage1 = addNoise(movingImage1);

 for i=1:360
     for j=1:512
         currentImage = rotateAndTranslate(movingImage1, i - 180 , j - 256);
         entropyGrid1(i, j) = jointEntropy(currentImage, originalImage1);
     end
     i
 end
 
 