function [jointTrajectory, Trajectory, timespan, occGrid] = test_case_C()
%Sample Manipulator and obstacle Trajectory 

%Number of desired steps
steps = 20;

%Range of motion for each joint
J2 = [-pi 3*pi];
J3 = [pi/4 3*pi/4];
J5 = [-pi/4 -1.1*pi/4];
J6 = [-pi/3 -1.1*pi/3];

%Joint position 2-7 in the array correspond to joint angles 1-6
jointTrajectory(2,:) = J2(1):(J2(2)-J2(1))/steps:J2(2);
jointTrajectory(3,:) = J3(1):(J3(2)-J3(1))/steps:J3(2);
jointTrajectory(5,:) = J5(1):(J5(2)-J5(1))/steps:J5(2);
jointTrajectory(6,:) = J6(1):(J6(2)-J6(1))/steps:J6(2);


%Joint position 1 in the array corresponds to the base (no angle change)
%Trajectory 1 stores time stamp information
%Trajectory 2 stores segment number
[r,c] = size(jointTrajectory(2,:));
jointTrajectory(1,:) = zeros(r,c);
jointTrajectory(7,:) = zeros(r,c);
jointTrajectory(4,:) = zeros(r,c);
timespan = 7;
Trajectory = zeros(2,c);
Trajectory(1,:) = 0:timespan/c:timespan-timespan/c;
Trajectory(2,:) = 1:1:c;

%--------------------------------------------------------------------
%Case for a 3-D parabolic trajectory
%--------------------------------------------------------------------
so = [1 0 0];                 %Initial Position (Use Interger Value) [X, Y, Z]
v = [-0.05 0 0.005];          %Initial Velocity [X, Y, Z]
a = [-0.2 -0.05 0.2];         %Constant Acceleration [X, Y, Z]
time = 0:0.01:10;             %Sampling Time: Later optimize based on v and a'
roll = 22.5;                  %Scaling factor for roll angle throughout trajectory
width = 1;                    %Scaling factor for ellipse width
height = 3;                   %Scaling factor for ellipse height in y direction
k = 0.025;                    %Scaling factor for cone expansion

occGrid = sample_obstacle(so,v,a,time,roll,width,height,k);

end