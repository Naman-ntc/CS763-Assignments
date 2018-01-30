function [ H ] = homography( p1, p2 )

%HOMOGRAPHY Summary of this function goes here

%Assuming points are matched and there are an equal number of points on both sides
%homography for p1->p2
numberOfPoints = size(p1, 2);
A = zeros(2*numberOfPoints, 9);
for i=1:numberOfPoints
	A(2*i - 1,1:2) = -p1(:,i)';
	A(2*i - 1, 3) = -1;
	A(2*i - 1, 4:6) = zeros(1, 3);
	A(2*i - 1, 7:8) = p2(1, i)*(p1(:, i)');
	A(2*i - 1, 9) = p2(1, i);
	A(2*i, 1:3) = zeros(1, 3);
	A(2*i, 4:5) = -p1(:, i)';
	A(2*i, 6) = -1;
	A(2*i, 7:8) = p2(2, i)*(p1(:, i)');
	A(2*i, 9) = p2(2, i);
end

[~,~,V] = svd(A);
P = V(:,9);
H = reshape(P,[3,3])';


%Detailed explanation goes here


end

