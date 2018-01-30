%% Rigit Transform between 2 sets of 3D Points

%% Load Data
%load('../input/Q1data.mat');

%% Your code here


%{
widthPoints = readPoints('../input/wembley.jpeg', 4);
dee1points = readPoints('../input/wembley.jpeg', 2);
dee2points = readPoints('../input/wembley.jpeg', 2);
verticalLinePoints = readPoints('../input/wembley.jpeg', 2);

save('Q2data','widthPoints','dee1points','dee2points','verticalLinePoints')
%}


load('Q2data');
crossRatioWidth = (norm(widthPoints(:,1) - widthPoints(:,3))/norm(widthPoints(:,1) - widthPoints(:,4)))/(norm(widthPoints(:,2) - widthPoints(:,3))/norm(widthPoints(:,2) - widthPoints(:,4)));
%If the distance between the edge and the closer end of the Dee (on the
%horizontal line) is x, then we get the following equation:

syms x

eqn = crossRatioWidth*(88*x + 44*44) - (x + 44)^2 == 0;
solution1 = solve(eqn, x, 'Maxdegree', 3);
widthSolutions = 2*double(solution1) + 44;
width = widthSolutions(2);


syms x1 y1 x2 y2 m1 c1 m2 c2  m3 c3   

eqns = [
    y1 == m1*x1 + c1,
    y2 == m2*x2 + c2,
    y1 == m3*x1 + c3,
    y2 == m3*x2 + c3,
    dee1points(2,1) == m1*dee1points(1,1) + c1,
    dee1points(2,2) == m1*dee1points(1,2) + c1,
    dee2points(2,1) == m2*dee2points(1,1) + c2,
    dee2points(2,2) == m2*dee2points(1,2) + c2,
    verticalLinePoints(2,1) == m3*verticalLinePoints(1,1) + c3,
    verticalLinePoints(2,2) == m3*verticalLinePoints(1,2) + c3
];

s = solve(eqns);
solution2 = double([s.x1, s.y1, s.x2, s.y2]);

verticalPoints = zeros(2, 4);
verticalPoints(:,1) = verticalLinePoints(:, 1);
verticalPoints(:,4) = verticalLinePoints(:, 2);
verticalPoints(:,2) = solution2(1:2)';
verticalPoints(:,3) = solution2(3:4)';

crossRatioLength = (norm(verticalPoints(:,1) - verticalPoints(:,3))/norm(verticalPoints(:,1) - verticalPoints(:,4)))/(norm(verticalPoints(:,2) - verticalPoints(:,3))/norm(verticalPoints(:,2) - verticalPoints(:,4)));

syms l

eqn = crossRatioLength*(36*l + l^2) - (l + 18)^2 == 0;
solution3 = solve(eqn, l, 'Maxdegree', 2);
lengthSolutions = double(solution3) + 18*2;
length = lengthSolutions(2);

width
length
