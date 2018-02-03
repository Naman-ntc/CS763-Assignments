%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               For Monument
image1 = imread('../input/monument/1.JPG');
image2 = imread('../input/monument/2.JPG');
[image_x,image_y,~] = size(image1);

panaroma_size = [1200,1500,3];
panaroma = zeros(panaroma_size);
panaroma(1:image_x,1:image_y,:) = image1;
H_1_to_1 = eye(3);
H_1_to_1(1,3) = 350;
%H_1_to_1(2,3) = 100;
%panaroma = add_img_to_panaroma(eye(3),panaroma,image1);

[f1,f2] = getFeaturePoints(image2,image1);
H_2_to_1 = ransacHomography(f1,f2,2);
panaroma = add_img_to_panaroma(H_2_to_1*H_1_to_1,panaroma,image2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
