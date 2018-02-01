[a,b] = getFeaturePoints('../input/hill/1.JPG','../input/hill/2.JPG');
ransacHomography(a,b,3);
