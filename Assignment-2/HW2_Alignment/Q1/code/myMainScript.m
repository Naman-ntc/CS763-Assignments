%% Rigit Transform between 2 sets of 3D Points

%{
p1 = readPoints('../input/wembley.jpeg',4);
q1 = [0,0,44,44;0,18,18,0];
field_edges = readPoints('../input/wembley.jpeg',3);
save('Q1data','p1','q1','field_edges');
%}

load('Q1data.mat');

H = homography(p1,q1);

new_field_edges = H * ([field_edges;[1,1,1]]);
for i = 1:3
    new_field_edges(:,i) = new_field_edges(:,i)/new_field_edges(3,i);
end

length = norm(new_field_edges(:,1)-new_field_edges(:,2));
width = norm(new_field_edges(:,2)-new_field_edges(:,3));

length
width