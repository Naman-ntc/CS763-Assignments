function [ val ] = myinterp2( f,a,b,c )
%MYINTERP2 Summary of this function goes here
%   Detailed explanation goes here
val = f(min(1,floor(a+0.5)),min(1,floor(b+0.5)));
end

