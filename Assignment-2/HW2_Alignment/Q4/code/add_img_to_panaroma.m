function [ panaroma ] = add_img_to_panaroma( H_2_to_1,panaroma,image2 )
%ADD_IMG_TO_PANAROMA Summary of this function goes here
%   Detailed explanation goes here
H_2_to_1 = H_2_to_1/H_2_to_1(3,3);
[image_x,image_y,~] = size(image2);

[bbox_x,bbox_y] = give_bbox(H_2_to_1,image_x,image_y);


for i=bbox_x
    for j=bbox_y
        orig = [i,j,1]/H_2_to_1;
        x_new = min(image_x,max(1,floor(0.5+orig(1))));
        y_new = min(image_y,max(1,floor(0.5+orig(2))));
        if (panaroma(i,j,1)==0)
            panaroma(i,j,1) = image2(x_new,y_new,1);
        else
            panaroma(i,j,1) = (image2(x_new,y_new,1)+ panaroma(i,j,1))/2;
        end  
        if (panaroma(i,j,2)==0)
            panaroma(i,j,2) = image2(x_new,y_new,2);
        else
            panaroma(i,j,2) = (image2(x_new,y_new,2)+ panaroma(i,j,2))/2;
        end  
        if (panaroma(i,j,3)==0)
            panaroma(i,j,3) = image2(x_new,y_new,3);
        else
            panaroma(i,j,3) = (image2(x_new,y_new,3)+ panaroma(i,j,3))/2;
        end  
        %[max(1,floor(orig(1)+0.5)),max(1,floor(orig(2)+0.5))] == [i,j]
    end
end

end

