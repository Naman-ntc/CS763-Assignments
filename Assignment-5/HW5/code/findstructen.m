function [ output ] = findstructen( image,point,Xgrads,Ygrads)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

output = zeros(6,6);

for i = point(1)-20:point(1)+20
    for j = point(2)-20:point(2)+20
        deltaI = [Xgrads(j,i) , Ygrads(j,i)];
        gradW = [[i,j,1,0,0,0]; [0,0,0,i,j,1]];
        temp = (double(deltaI) * double(gradW));
        output = output + transpose(temp) * temp;
    end
end

end

