clear;
close all;
clc;

%% Your Code Here
%
%
%
%
%
%
%
%


%% Save all the trajectories frame by frame
% variable trackedPoints assumes that you have an array of size 
% No of frames * 2(x, y) * No Of Features
% noOfFeatures is the number of features you are tracking
% Frames is array of all the frames(assumes grayscale)
noOfPoints = 1;
for i=1:N
    NextFrame = Frames(i,:,:);
    imshow(uint8(NextFrame)); hold on;
    for nF = 1:noOfFeatures
        plot(trackedPoints(1:noOfPoints, 1, nF), trackedPoints(1:noOfPoints, 2, nF),'*')
    end
    hold off;
    saveas(gcf,strcat('output/',num2str(i),'.jpg'));
    close all;
    noOfPoints = noOfPoints + 1;
end 
   
