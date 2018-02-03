function [matched_pts1, matched_pts2] = get_matchedPoints(img1, img2)
%GET_MATCHEDPOINTS Summary of this function goes here
%   Detailed explanation goes here
    gImg1 = rgb2gray(img1);
    points1 = detectSURFFeatures(gImg1);
    [features1, points1] = extractFeatures(gImg1, points1);
    
    gImg2 = rgb2gray(img2);
    points2 = detectSURFFeatures(gImg2);
    [features2, points2] = extractFeatures(gImg2, points2);
    
    indexPairs = matchFeatures(features2, features1, 'Unique', true);

    matchedPoints2 = points2(indexPairs(:,1), :);
    matchedPoints1 = points1(indexPairs(:,2), :);
    
    matched_pts1 = matchedPoints1.Location;
    matched_pts2 = matchedPoints2.Location;
end

