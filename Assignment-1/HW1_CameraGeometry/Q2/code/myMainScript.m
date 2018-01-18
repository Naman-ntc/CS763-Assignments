image = imread('../input/rad_checkerbox.jpg');
image = im2double(rgb2gray(image));

q1 = 1;
q2 = 0.5;
epsilon = 1e-3;
[X,Y,Z] = size(image);
half_Y = Y/2;
half_X = X/2;

A = zeros(size(image));

for i = 1:X
    for j = 1:Z
        x = (i - half_X);
        y = (j - half_Y);
        x = x/(half_X); y = y/(half_Y);
        r = sqrt(x^2 + y^2);
        q =  1 + q1*r + q2*r*r;
        src_x = x/q;
        src_y = y/q; 
        A(i,j) = interp2(image,half_X*(1+src_x),half_Y*(1+src_y));
    end
end