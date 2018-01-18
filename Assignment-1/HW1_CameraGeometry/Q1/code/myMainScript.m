% Data Collection

%image = imread('../data/input.jpg');
%[X,Y,Z] = size(image);
%N = 6;
%image_points = readPoints(image,N); % We can select more points as well (Ziesserman says 28!) 
%object_points = zeros([3,N]); % Manually add point locations of 3D space
%save('locations','image_points','object_points');

% These are Euclidean coordinates!!

% End Data Collection

load('locations.mat')
N = size(image_points);
N = N(2);
image_points = [image_points; ones(1,N)];
object_points = [object_points; ones(1,N)];
% XXXXXXXXXXXXXXXXXXX     TRANSFORM THE COORDINATES LEFT
A = zeros(2*N,12);
for i = 1:N
    A(2*i-1,1:4) = 0;
    A(2*i-1,5:8) = -1*image_points(3,i)*object_points(:,i);
    A(2*i-1,9:12) = image_points(2,i)*object_points(:,i);
    A(2*i,1:4) = image_points(3,i)*object_points(:,i);
    A(2*i,5:8) = 0;
    A(2*i,9:12) = -1*image_points(1,i)*object_points(:,i);
end

[~,~,V] = svd(A);
P = V(:,12); % |P| not exactly 1 but close
P = reshape(P,[3,4]); 

[~,~,V] = svd(P); % Can also be computed algebraically actually (Ziesermann pg150)
X0 = V(:,4); % Check -- Last homogeneous coordinated almost 1 but not exactly
X0 = X0(1:3); 

% P = [M | -MC]

M = P(1:3,1:3);
[R,M] = qr(M);
% R is the rotation matrix and M is a upper triangular matrix
