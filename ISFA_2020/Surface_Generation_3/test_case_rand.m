function [jointTrajectory, Trajectory, timespan, occGrid] = test_case_A()
%Sample Manipulator and obstacle Trajectory 

%Number of desired steps
steps = 5;
timespan = 3;

%Use this to set the direction of random motion
dir = [0 1 1 0 0 1 1];

%Range of motion for each joint
for ii = 1:100
    for jj = 2:7
        if dir(jj) == 1
            J(jj,ii) = 0 - rand;
        else
            J(jj,ii) = 0 + rand;
        end
    end
end

%Random joint velocities
stepSpeed = zeros(1,100);
for jj = 1:100
    stepSpeed(jj) = rand;
end

%Normalize speeds to define spacing throughout trajectory
stepSpeed = stepSpeed/sum(stepSpeed);

for ii = 1:steps+1
    %Joint position 2-7 in the array correspond to joint angles 1-6
    jointTrajectorys(2,ii) = J2(1) + (J2(2)-J2(1))*sum(stepSpeed(1:ii));
    jointTrajectorys(3,ii) = (J3(2)-J3(1))*sum(stepSpeed(1:ii));
    jointTrajectorys(4,ii) = (J4(2)-J4(1))*sum(stepSpeed(1:ii));
    jointTrajectorys(5,ii) = (J5(2)-J5(1))*sum(stepSpeed(1:ii));
    jointTrajectorys(6,ii) = (J6(2)-J6(1))*sum(stepSpeed(1:ii));
    jointTrajectorys(7,ii) = (J7(2)-J7(1))*sum(stepSpeed(1:ii));
end

%Joint position 2-7 in the array correspond to joint angles 1-6
jointTrajectory(2,:) = J2(1):(J2(2)-J2(1))/steps:J2(2);
jointTrajectory(3,:) = J3(1):(J3(2)-J3(1))/steps:J3(2);
jointTrajectory(4,:) = J4(1):(J4(2)-J4(1))/steps:J4(2);   
jointTrajectory(5,:) = J5(1):(J5(2)-J5(1))/steps:J5(2);   
jointTrajectory(6,:) = J6(1):(J6(2)-J6(1))/steps:J6(2);
jointTrajectory(7,:) = J7(1):(J7(2)-J7(1))/steps:J7(2);

%Joint position 1 in the array corresponds to the base (no angle change)
%Trajectory 1 stores time stamp information
%Trajectory 2 stores segment number
[r,c] = size(jointTrajectory(2,:));
jointTrajectory(1,:) = zeros(r,c);
Trajectory = zeros(2,c);
Trajectory(1,:) = 0:timespan/c:timespan-timespan/c;
Trajectory(2,:) = 1:1:c;


%--------------------------------------------------------------------
%Case for a 3-D parabolic trajectory
%--------------------------------------------------------------------
so = [-1 -0.2 0.2];           %Initial Position (Use Interger Value) [X, Y, Z]
v = [0.35 -0.05 -0.005];      %Initial Velocity [X, Y, Z]
a = [0.2 -0.05 0.04];         %Constant Acceleration [X, Y, Z]
time = 0:0.01:10;             %Sampling Time: Later optimize based on v and a'
roll = 22.5;                  %Scaling factor for roll angle throughout trajectory
width = 1;                    %Scaling factor for ellipse width
height = 3;                   %Scaling factor for ellipse height in y direction
k = 0.025;                    %Scaling factor for cone expansion

occGrid = sample_obstacle(so,v,a,time,roll,width,height,k);

end

