function [ panaroma ] = add_img_to_panaroma( H_2_to_1,panaroma,image2 )
%ADD_IMG_TO_PANAROMA Summary of this function goes here
%   Detailed explanation goes here
H_2_to_1 = H_2_to_1/H_2_to_1(3,3);
[image_x,image_y,~] = size(image2);

[bbox_x,bbox_y] = give_bbox(H_2_to_1,image_x,image_y);

size(bbox_x)
size(bbox_y)
for i=bbox_x
    i
    for j=bbox_y
        orig = [i,j,1]/H_2_to_1;
        orig = orig/orig(3);
        if (panaroma(i,j,1)==0)
            panaroma(i,j,1) = interp2(image2(:,:,1),orig(1),orig(2),'nearest');
        else
            panaroma(i,j,1) = (interp2(image2(:,:,1),orig(1),orig(2),'nearest') + panaroma(i,j,1))/2;
        end     
    end
end

end

