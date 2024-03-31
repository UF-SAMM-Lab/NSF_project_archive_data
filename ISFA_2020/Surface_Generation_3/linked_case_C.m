function [jointTrajectoryR, TrajectoryR, jointTrajectoryL, TrajectoryL, timespan] = linked_case_C()


%Number of desired steps
steps = 5;
%Duration of procedure
timespan = 7;

%Range of motion for right arm
J1R = [0 pi/4];
J2R = [0 pi/3];
J3R = [0 0.0001];
J4R = [0 -pi/2];
J5R = [0 -pi/4];

%Range of motion for left arm
J1L = J1R; %Shared joint, must be the same
J2L = [0 -pi/3];
J3L = [0 0.0001];
J4L = [0 pi/2];
J5L = [0 -pi/4];

%Vector for Right arm trajectory
jointTrajectoryR(1,:) = J1R(1):(J1R(2)-J1R(1))/steps:J1R(2);
jointTrajectoryR(2,:) = J2R(1):(J2R(2)-J2R(1))/steps:J2R(2);
jointTrajectoryR(3,:) = J3R(1):(J3R(2)-J3R(1))/steps:J3R(2);
jointTrajectoryR(4,:) = J4R(1):(J4R(2)-J4R(1))/steps:J4R(2);
jointTrajectoryR(5,:) = J5R(1):(J5R(2)-J5R(1))/steps:J5R(2);

%Vector for Left arm trajectory
jointTrajectoryL(1,:) = J1L(1):(J1L(2)-J1L(1))/steps:J1L(2);
jointTrajectoryL(2,:) = J2L(1):(J2L(2)-J2L(1))/steps:J2L(2);
jointTrajectoryL(3,:) = J3L(1):(J3L(2)-J3L(1))/steps:J3L(2);
jointTrajectoryL(4,:) = J4L(1):(J4L(2)-J4L(1))/steps:J4L(2);
jointTrajectoryL(5,:) = J5L(1):(J5L(2)-J5L(1))/steps:J5L(2);

%Trajectory 1 stores time stamp information
%Trajectory 2 stores segment number

%Right arm
[~,c] = size(jointTrajectoryR(2,:));
TrajectoryR = zeros(2,c);
TrajectoryR(1,:) = 0:timespan/c:timespan-timespan/c;
TrajectoryR(2,:) = 1:1:c;

%Left arm
[~,c] = size(jointTrajectoryL(2,:));
TrajectoryL = zeros(2,c);
TrajectoryL(1,:) = 0:timespan/c:timespan-timespan/c;
TrajectoryL(2,:) = 1:1:c;

end