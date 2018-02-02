%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               For Monument
image1 = imread('../input/monument/1.JPG');
image2 = imread('../input/monument/2.JPG');

panaroma_size = [1200,1500,3];
panaroma = zeros(panaroma_size);
%panaroma = add_img_to_panaroma(eye(3),panaroma,image1);
%imshow(panaroma)

[f1,f2] = getFeaturePoints(image2,image1);
H_2_to_1 = ransacHomography(f1,f2,3);
[image_x,image_y,~] = size(image1);
panaroma = add_img_to_panaroma(H_2_to_1,panaroma,image2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
