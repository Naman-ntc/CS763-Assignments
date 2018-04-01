function [ output ] = find2dstructen( image,point,Xgrads,Ygrads)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

output = zeros(2,2);

for i = point(1)-20:point(1)+20
    for j = point(2)-20:point(2)+20
        deltaI = [Xgrads(i,j) , Ygrads(i,j)];
        temp = double(deltaI);
        output = output + transpose(temp) * temp;
    end
end

end