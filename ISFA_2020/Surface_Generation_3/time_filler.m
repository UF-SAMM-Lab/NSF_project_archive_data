function [time] = time_filler(time,b)
%--------------------------------------------------------------------------
%The purpose of this function is to linearly interpolate between two three
%dimensional points and specify "b" equally spaced values between them

%INPUT: curve: The endpoints of the line that is to be linearly estimated
%       b: specification for the number of points desired between endpoints

%OUTPUT: curve: Line between the two input points
%--------------------------------------------------------------------------

t1 = time(1);
t2 = time(end);


tspace = (t2 - t1)/(b-1);

    for ii = 2:b-1
        time(1,ii) = time(1,1) + tspace*(ii-1);
    end
end