function [curve] = point_filler(curve,b)
%--------------------------------------------------------------------------
%The purpose of this function is to linearly interpolate between two three
%dimensional points and specify "b" equally spaced values between them

%INPUT: curve: The endpoints of the line that is to be linearly estimated
%       b: specification for the number of points desired between endpoints

%OUTPUT: curve: Line between the two input points
%--------------------------------------------------------------------------

x1 = curve(1,1,1);
y1 = curve(1,1,2);
z1 = curve(1,1,3);
x2 = curve(end,1,1);
y2 = curve(end,1,2);
z2 = curve(end,1,3);

xspace = (x2 - x1)/(b-1);
yspace = (y2 - y1)/(b-1);
zspace = (z2 - z1)/(b-1);

    for ii = 2:b-1
        curve(ii,1,1) = curve(1,1,1) + xspace*(ii-1);
        curve(ii,1,2) = curve(1,1,2) + yspace*(ii-1);
        curve(ii,1,3) = curve(1,1,3) + zspace*(ii-1);
    end
end

