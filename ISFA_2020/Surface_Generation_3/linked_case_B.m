function [jointTrajectoryR, TrajectoryR, jointTrajectoryL, TrajectoryL, jointTrajectoryC, TrajectoryC, timespan] = linked_case_B()


%Number of desired steps
steps = 10;
%Duration of procedure
timespan = 6;

%Range of motion for right arm
J1Ra = [0 pi/4];
J1Rb = [0 0.0001];
J1Rc = [0 0.0001];
J3Ra = [0 0.0001];
J3Rb = [0 0.0001];
J3Rc = [0 0.0001];
J4R =  [0 0.0001];

%Range of motion for left arm
J1La = J1Ra; %Shared joint, must be the same
J1Lb = J1Rb;
J1Lc = J1Rc;
J3La = [0 0.0001];
J3Lb = [0 0.0001];
J3Lc = [0 0.0001];
J4L =  [0 0.0001];

%Range of motion for core
J1Ca = J1Ra; %Shared joint, must be the same
J1Cb = J1Rb;
J1Cc = J1Rc;
J2Ca = [0 0.0001];
J2Cb = [0 0.0001];
J2Cc = [0 0.0001];

%Vector for Right arm trajectory
jointTrajectoryR(1,:) = J1Ra(1):(J1Ra(2)-J1Ra(1))/steps:J1Ra(2);
jointTrajectoryR(2,:) = J1Rb(1):(J1Rb(2)-J1Rb(1))/steps:J1Rb(2);
jointTrajectoryR(3,:) = J1Rc(1):(J1Rc(2)-J1Rc(1))/steps:J1Rc(2);
jointTrajectoryR(4,:) = J3Ra(1):(J3Ra(2)-J3Ra(1))/steps:J3Ra(2);
jointTrajectoryR(5,:) = J3Rb(1):(J3Rb(2)-J3Rb(1))/steps:J3Rb(2);
jointTrajectoryR(6,:) = J3Rc(1):(J3Rc(2)-J3Rc(1))/steps:J3Rc(2);
jointTrajectoryR(7,:) = J4R(1):(J4R(2)-J4R(1))/steps:J4R(2);

%Vector for Left arm trajectory
jointTrajectoryL(1,:) = J1La(1):(J1La(2)-J1La(1))/steps:J1La(2);
jointTrajectoryL(2,:) = J1Lb(1):(J1Lb(2)-J1Lb(1))/steps:J1Lb(2);
jointTrajectoryL(3,:) = J1Lc(1):(J1Lc(2)-J1Lc(1))/steps:J1Lc(2);
jointTrajectoryL(4,:) = J3La(1):(J3La(2)-J3La(1))/steps:J3La(2);
jointTrajectoryL(5,:) = J3Lb(1):(J3Lb(2)-J3Lb(1))/steps:J3Lb(2);
jointTrajectoryL(6,:) = J3Lc(1):(J3Lc(2)-J3Lc(1))/steps:J3Lc(2);
jointTrajectoryL(7,:) = J4L(1):(J4L(2)-J4L(1))/steps:J4L(2);

%Vector for Left arm trajectory
jointTrajectoryC(1,:) = J1Ca(1):(J1Ca(2)-J1Ca(1))/steps:J1Ca(2);
jointTrajectoryC(2,:) = J1Cb(1):(J1Cb(2)-J1Cb(1))/steps:J1Cb(2);
jointTrajectoryC(3,:) = J1Cc(1):(J1Cc(2)-J1Cc(1))/steps:J1Cc(2);
jointTrajectoryC(4,:) = J2Ca(1):(J2Ca(2)-J2Ca(1))/steps:J2Ca(2);
jointTrajectoryC(5,:) = J2Cb(1):(J2Cb(2)-J2Cb(1))/steps:J2Cb(2);
jointTrajectoryC(6,:) = J2Cc(1):(J2Cc(2)-J2Cc(1))/steps:J2Cc(2);

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

%Core
[~,c] = size(jointTrajectoryL(2,:));
TrajectoryC = zeros(2,c);
TrajectoryC(1,:) = 0:timespan/c:timespan-timespan/c;
TrajectoryC(2,:) = 1:1:c;

end