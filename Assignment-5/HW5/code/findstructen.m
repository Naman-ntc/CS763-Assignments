function [ output ] = findstructen( image,point,Xgrads,Ygrads)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
point
output = zeros(6,6);

bbox = 81;
mp = floor(bbox/2);

for i = point(1)-mp:point(1)+mp
    for j = point(2)-mp:point(2)+mp
        deltaI = [Xgrads(i,j) , Ygrads(i,j)];
        gradW = [[j,i,1,0,0,0]; [0,0,0,j,i,1]];
        temp = (double(deltaI) * double(gradW));
        output = output + transpose(temp) * temp;
    end
end

end

