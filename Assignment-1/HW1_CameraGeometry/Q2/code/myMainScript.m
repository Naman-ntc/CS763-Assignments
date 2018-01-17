%% MyMainScript

%tic;
%% Your code here
image = imread('../input/rad_checkerbox.jpg');
image = im2double(rgb2gray(image));
%m1 = min(image(:));
%image = image - m1;
%m2 = max(image(:));
%image = image/m2;
%image = (image - 0.5)*2;
%imshow(image,[]);
q1 = 1;
q2 = 0.5;
epsilon = 1e-3;
[X,Y,Z] = size(image);
half_Y = Y/2;
half_X = X/2;
A = zeros(size(image));
for i = 1:X
    for j = 1:Y 
        x = (i - half_X);
        y = (j - half_Y);
        x = x/(20*half_X); y = y/(20*half_Y);
        r = sqrt(x^2 + y^2);
        q =  1 - q1*r - q2*r*r;
        %q
        %x_u = [i , j ,1];
        %D = [[1,0,q],[0,1,q],[0,0,1]];
        x_u0 = (i-half_X)/q;
        y_u0 = (j-half_Y)/q;
        x_u_i1 = x_u0;
        y_u_i1 = y_u0;
        for k = 1:300
            %x = (x_u_i1 - half_X);
            %y = (y_u_i1 - half_Y);
            x = x_u_i1/(20*half_X); y = y_u_i1/(20*half_Y);
            r = sqrt(x^2 + y^2);
            q =  1 - q1*r - q2*r*r;
            x_u_i1 = (i-half_X)/q;
            y_u_i1 = (j-half_Y)/q;
        end
        x_u_i1 = x_u_i1+half_X;
        y_u_i1 = y_u_i1+half_Y;
        x_u_i1 = max(min(x_u_i1,X),1);
        y_u_i1 = max(min(y_u_i1,Y),1);
        A(round(x_u_i1), round(y_u_i1)) = image(i,j);
    end
end
%toc;
