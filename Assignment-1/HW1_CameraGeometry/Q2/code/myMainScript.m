%% MyMainScript

tic;
%% Your code here
Img = im2double(imread('../input/rad_checkerbox.jpg'));
Img = rgb2gray(Img);
q2 = 0.5;
q1 = 1;
sz = size(Img);
A = zeros(sz(1), sz(2));
a = round(sz/2);
cx=a(1);
cy=a(2);

H_inv_mat = eye(3);
H_mat = eye(3);

for xd = 1:sz(1)
    for yd = 1:sz(2)
        norml_Xd = [(xd-cx)/cx, (yd-cy)/cy, 1]';
        xu = (xd-cx)/cx;
        yu = (yd-cy)/cy;
        xPrev = 10000;
        yPrev = 10000;
        while norm([xu-xPrev, yu-yPrev]) > 0.001 %run until convergence
        %for i =1:4 %works in 4 iterations as well
            xPrev = xu;
            yPrev = yu;
            %recentering
            H_mat = eye(3); %The matrix we have to multiply x_d with it to get the next guess for x_u
            r_vec = [xu, yu];
            r = norm(r_vec); % calculate r using the latest guesses for x_u
            q_fac = 1 + (q1*r)+(q2*(r^2));
            f = 1/q_fac - 1; %Delta(x) = x_d*f and Delta(y) = y_d*f
            H_mat(1,3) = norml_Xd(1)*f;
            H_mat(2,3) = norml_Xd(2)*f; % complete the matrix that is to be multiplied
            %H_inv_mat = inv(H_mat);
            T = H_mat*norml_Xd; % T is our guess for x_u in the homogenous coods with third entry 1
            xu = T(1);
            yu = T(2); %Take first two values of T and assign to x_u
             end
        
        A(max(1,min(round(xu*cx+cx),sz(1))), max(1,min(round(yu*cy+cy), sz(2)))) = Img(xd, yd); %Assign to the nearest inegral index
    end
end
 imshow(A, []) %display the undistorted image

% CODE TO REGENERATE THE DISTORTED IMAGE USING THE GIVEN TRANSFORMATION 
% 
% R = zeros(size(A));
% for i=1:size(A, 1)
%     for j=1:size(A, 2)
%         idash = (i-cx)/cx;
%         jdash = (j-cy)/cy;
%         r = norm([idash, jdash]');
%         fac = 1 + q1*r + q2*r*r;
%         xd = idash*fac*cx + cx;
%         yd = jdash*fac*cy + cy;
%         R(max(1, min(round(xd), sz(1))), max(1, min(round(yd), sz(2)))) = A(i, j);
%     end
% i    
% end
% toc;
