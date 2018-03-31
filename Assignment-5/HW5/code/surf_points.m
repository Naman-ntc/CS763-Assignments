function [ points ] = surf_points( image,numFeatures )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

points = detectSURFFeatures(image);
points = round(points.Location);

filterX = [[-1,0,1]; [-2,0,2]; [-1,0,1]]/8;
filterY = transpose(filterX);

imgradX = imfilter(image,filterX);
imgradY = imfilter(image,filterY);

countPoints = size(points);
eigenvalues = zeros(countPoints);

for i = 1:countPoints
    if (points(i,1) <21 || points(i,2) <21 || points(i,1) > 619 || points(i,2) > 459)
        eigenvalues(i) = -1;
    else
        temp = eig(findstructen(image,points(i,:),imgradX,imgradY));
        eigenvalues(i) = temp(1);
    end
end

[~,eigorder] = sort(eigenvalues,'descend');
points = points(eigorder,:);

points = (points(1:numFeatures,:));
end