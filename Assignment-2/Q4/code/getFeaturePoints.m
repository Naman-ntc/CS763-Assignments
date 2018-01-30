function [ matchedPoints1, matchedPoints2 ] = getFeauturePoints( s1, s2 )

    image1 = imread(s1);
    image1 = rgb2gray(image1);
    image2 = imread(s2);
    image2 = rgb2gray(image2);


    points1 = detectSURFFeatures(image1);
    points2 = detectSURFFeatures(image2);

    [f1, vpts1] = extractFeatures(image1, points1);
    [f2, vpts2] = extractFeatures(image2, points2);

    indexPairs = matchFeatures(f1, f2);

    matchedPoints1 = vpts1(indexPairs(:, 1));
    matchedPoints2 = vpts2(indexPairs(:, 2));
    
    matchedPoints1 = matchedPoints1.Location;
    matchedPoints2 = matchedPoints2.Location;
end