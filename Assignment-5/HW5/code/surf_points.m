function [ points ] = surf_points( image,numFeatures )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

points = detectSURFFeatures(image);
points = fliplr(round(points.Location));

% filterX = [[-1,0,1]; [-2,0,2]; [-1,0,1]]/8;
% filterY = transpose(filterX);

bbox = 81;

[imgradX,imgradY] = imgradientxy(image);

countPoints = size(points);
eigenvalues = zeros(countPoints);

for i = 1:countPoints
    if (points(i,2) <bbox/2 || points(i,1) <bbox/2 || points(i,2) > 480 || points(i,1) > 430)
        eigenvalues(i) = -1;
    else
        temp = eig(find2dstructen(image,points(i,:),imgradX,imgradY));
        eigenvalues(i) = temp(1);
    end
end

[~,eigorder] = sort(eigenvalues,'descend');
points = points(eigorder,:);

points = (points(1:numFeatures,:));
end