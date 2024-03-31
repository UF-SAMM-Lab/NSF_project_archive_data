function [jointTrajectory, Trajectory, timespan, offset] = just_robot_test_case_random()
%Sample Manipulator and obstacle Trajectory 

%Set desired offset value for each link on edo
offset = [0.07 0.07 0.07 0.07 0.07 0.07 0.07];

%Number of desired steps
steps = 5;

%Range of motion for each joint
J2 = [-pi*rand      pi*rand];   %Limits on Joint 1: -pi to pi
J3 = [-pi/2*rand  pi/2*rand];   %Limits on Joint 1: -pi/2 to pi/2
J4 = [-pi/2*rand  pi/2*rand];   %Limits on Joint 1: -pi/2 to pi/2
J5 = [-pi*rand     -pi*rand];   %Limits on Joint 1: -pi to pi
J6 = [-pi/2*rand  pi/2*rand];   %Limits on Joint 1: -pi to pi

%Joint position 2-7 in the array correspond to joint angles 1-6
jointTrajectory(2,:) = J2(1):(J2(2)-J2(1))/steps:J2(2);
jointTrajectory(3,:) = J3(1):(J3(2)-J3(1))/steps:J3(2);
jointTrajectory(4,:) = J4(1):(J4(2)-J4(1))/steps:J4(2);   
jointTrajectory(5,:) = J5(1):(J5(2)-J5(1))/steps:J5(2);   
jointTrajectory(6,:) = J6(1):(J6(2)-J6(1))/steps:J6(2);
jointTrajectory(7,:) = zeros(size(jointTrajectory(6,:)));

%Joint position 1 in the array corresponds to the base (no angle change)
%Trajectory 1 stores time stamp information
%Trajectory 2 stores segment number
[r,c] = size(jointTrajectory(2,:));
jointTrajectory(1,:) = zeros(r,c);
timespan = 7;
Trajectory = zeros(2,c);
Trajectory(1,:) = 0:timespan/c:timespan-timespan/c;
Trajectory(2,:) = 1:1:c;

end