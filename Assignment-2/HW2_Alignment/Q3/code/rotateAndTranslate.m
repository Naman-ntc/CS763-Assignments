function [warpedImage] = rotateAndTranslate(image, angle, tx)


if size(size(image), 2) == 3
   image = rgb2gray(image);
end

%{
[h, l] = size(image);
midH = h/2;
midL = l/2;
angle = -angle*pi/180;
rotationMatrix = [cos(angle), -sin(angle); sin(angle), cos(angle)];
warpedImage = zeros(h, l);
for i=1:h
    for j=1:l
        source = rotationMatrix*[i - midH ;j - midL ];
        source(1) = source(1) - tx;
        source = source + [midH;midL];
        warpedImage(i, j) = interp2(image, source(2), source(1));
    end
end
warpedImage(isnan(warpedImage)) = 0;
%}

warpedImage = imrotate(image, angle, 'crop');
warpedImage = imtranslate(warpedImage, [tx, 0,]);
warpedImage = max(warpedImage, 0);

end