function [ H ] = ransacHomography( x1, x2, thresh )
%RANSACHOMOGRAPHY Summary of this function goes here
%   Detailed explanation goes here
N = size(x1,1);
x1_proj = [x1, ones(N,1)];
x2_proj = [x2, ones(N,1)];
iter = 30;
arr = zeros(iter,1);
Hrr = zeros(iter,3,3);
for i = 1:iter
    indices = randperm(N,4);
    H = homography(x1(indices,:)',x2(indices,:)');
    temp = (x1_proj)*(H');
    temp = temp./repmat(temp(:,3),1,3);
    loss = sum((x2_proj - temp).^2,2);
    count = sum(loss<thresh);
    arr(i) = count;
    Hrr(i,:,:) = H;
end
[~,index] = max(arr);
H = Hrr(index,:,:);
end

