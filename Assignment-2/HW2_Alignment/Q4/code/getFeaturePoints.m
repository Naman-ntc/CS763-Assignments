function [ matchedPoints1, matchedPoints2 ] = getFeauturePoints( image1, image2 )

    image1 = rgb2gray(image1);
    image2 = rgb2gray(image2);

    points1 = detectSURFFeatures(image1);
    points2 = detectSURFFeatures(image2);

    [f1, vpts1] = extractFeatures(image1, points1);
    [f2, vpts2] = extractFeatures(image2, points2);

    indexPairs = matchFeatures(f1, f2);

    matchedPoints1 = vpts1(indexPairs(:, 1));
    matchedPoints2 = vpts2(indexPairs(:, 2));
    
    %figure; showMatchedFeatures(image1,image2,matchedPoints1,matchedPoints2);
    %legend('matched points 1','matched points 2');
    
    matchedPoints1 = matchedPoints1.Location;
    matchedPoints2 = matchedPoints2.Location;
    
    matchedPoints1 = circshift(matchedPoints1,1,2);
    matchedPoints2 = circshift(matchedPoints2,1,2);
    
    
    %{
    Before circshift it was this :
    
    ans =
    
    1.0e+03 *
    
    [1.0475    0.7901]
    
    
    ans =

   1.0e+03 *

    [1.1179    0.8178]
    
    %}
    
end