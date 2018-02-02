function [ bbox_x,bbox_y ] = give_bbox( H,x,y )
%GIVE_BBOX Summary of this function goes here
%   Detailed explanation goes here
A = H*[0,0,1]';
B = H*[x,0,1]';
C = H*[0,y,1]';
D = H*[x,y,1]';


[A(2),B(2),C(2),D(2)]
[A(1),B(1),C(1),D(1)]
bbox_y = [floor(min([A(2),B(2),C(2),D(2)]))+1:ceil(max([A(2),B(2),C(2),D(2)]))];
bbox_x = [floor(min([A(1),B(1),C(1),D(1)]))+1:ceil(max([A(1),B(1),C(1),D(1)]))];
end