clc;
clear;
o = 200;
test_time = zeros(1,o);
for mm = 1:o
    %--------------------------------------------------------------------------
%This is all assumed to be given: The joint trajectory of the robot and the
%joint locations of the human throughout time

%Sample trajectory info (joint angles thoroughout time)
[jointTrajectory, Trajectory, timespan, safety_factor] = just_robot_test_case_random();
%Sample obstacle info
[jointTrajectory_obstacle, Trajectory_obstacle] = full_human_joints();

%Determine XYZ position of each joint at each pose
a = 22;  %Number of points used to define human
[~,b] = size(jointTrajectory_obstacle);
points_obstacle = zeros(4,a,b);

%Right arm (use 1 in function)
for ii = 1:b
    points_obstacle(:,:,ii) = body(jointTrajectory_obstacle(:,ii),a);
end
%--------------------------------------------------------------------------


visual_sweep = 0;         %Displays the surface sweep
visual_obstacle = 0;      %Displays the obstacal with the surface sweep
visual_robot = 0;         %Displays the robot along the surface sweep
visual_object_out = 0;    %Displays the point cloud of collision points
visual_time_compare = 0;  %Visualization clarifying the time intersection


%Set start time for computation
t = cputime;

%Determine XYZ position of each joint at each pose
a = 12;  %Number of points used to define human
[~,b] = size(jointTrajectory);
points = zeros(4,a,b);
for ii = 1:b
    points(:,:,ii) = edo_points(jointTrajectory(:,ii),a);
end

%Set grid spacing for the collision check
grid_spacing = 0.05; %meters between each point in the standard grid

%Determine the area swept out. Data stored in XYZ coordinates within patches
start = 2;
patches = generate_surface(points,visual_sweep,Trajectory,start,grid_spacing);

%Create articulated obstacles to interact with the robot's sweep
start = 2;
human = generate_surface(points_obstacle,visual_sweep,Trajectory_obstacle,start,grid_spacing);

%Check for collisions, output affected sements
[collision, segments, obstacle] = collision_check(patches,human,timespan,visual_sweep,grid_spacing);

%Visualizations: 
if visual_robot || visual_sweep || visual_obstacle || visual_object_out || visual_time_compare
    visualize(human,collision,visual_robot,visual_sweep,visual_obstacle,jointTrajectory,visual_object_out,obstacle,visual_time_compare,patches);
end

%Determine elapsed cpu time
e = cputime-t;
fprintf('Total computation time: %g\n',e);
    test_time(mm) = e;
end
fprintf('Average Test Time for A over %i Tests: %f',o,mean(test_time(length(test_time)/2:end)))

