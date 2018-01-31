function [warpedImage] = rotateAndTranslate(s)

image = imread(s);
if size(size(image), 2) == 3
   image = rgb2gray(image);
end
[h, l] = size(image);
midH = h/2;
midL = l/2;
angle = -23.5*pi/180;
rotationMatrix = [cos(angle), -sin(angle); sin(angle), cos(angle)];
warpedImage = zeros(h, l);
for i=1:h
    for j=1:l
        source = rotationMatrix*[i - midH ;j - midL ];
        source(1) = source(1) + 3;
        source = source + [midH;midL];
        warpedImage(i, j) = interp2(image, source(2), source(1));
    end
end

end