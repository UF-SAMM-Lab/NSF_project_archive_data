function [jointTrajectory, Trajectory, timespan] = full_human_joints()

%Number of desired steps
steps = 6;
%Duration of procedure
timespan = 8;

%Joint angles torso
J1a = [0 0.0001];
J1b = [0 0.0001];
J1c = [0 0.0001];

%Joint angles right arm
J2a = [0 0.0001];
J2b = [0 0.0001];
J2c = [0 0.0001];
J3 =  [0 0.0001];

%Joint angles left arm
J5a = [0 0.0001];
J5b = [0 0.0001];
J5c = [0 0.0001];
J6 =  [0 0.0001];

%Joint angles neck
J8a = [0 0.0001];
J8b = [0 0.0001];
J8c = [0 pi];

%Vector for joint angles
jointTrajectory(1,:)  = J1a(1):(J1a(2)-J1a(1))/steps:J1a(2);
jointTrajectory(2,:)  = J1b(1):(J1b(2)-J1b(1))/steps:J1b(2);
jointTrajectory(3,:)  = J1c(1):(J1c(2)-J1c(1))/steps:J1c(2);
jointTrajectory(4,:)  = J2a(1):(J2a(2)-J2a(1))/steps:J2a(2);
jointTrajectory(5,:)  = J2b(1):(J2b(2)-J2b(1))/steps:J2b(2);
jointTrajectory(6,:)  = J2c(1):(J2c(2)-J2c(1))/steps:J2c(2);
jointTrajectory(7,:)  = J3(1):(J3(2)-J3(1))/steps:J3(2);
jointTrajectory(8,:)  = J5a(1):(J5a(2)-J5a(1))/steps:J5a(2);
jointTrajectory(9,:)  = J5b(1):(J5b(2)-J5b(1))/steps:J5b(2);
jointTrajectory(10,:) = J5c(1):(J5c(2)-J5c(1))/steps:J5c(2);
jointTrajectory(11,:) = J6(1):(J6(2)-J6(1))/steps:J6(2);
jointTrajectory(12,:) = J8a(1):(J8a(2)-J8a(1))/steps:J8a(2);
jointTrajectory(13,:) = J8b(1):(J8b(2)-J8b(1))/steps:J8b(2);
jointTrajectory(14,:) = J8c(1):(J8c(2)-J8c(1))/steps:J8c(2);

%Trajectory 1 stores time stamp information
%Trajectory 2 stores segment number

%Right arm
[~,c] = size(jointTrajectory(2,:));
Trajectory = zeros(2,c);
Trajectory(1,:) = 0:timespan/c:timespan-timespan/c;
Trajectory(2,:) = 1:1:c;


end