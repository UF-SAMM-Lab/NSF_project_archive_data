clc;
clear;
close all;

visual_sweep = 1;
grid_spacing = 0.05; 

%Sample obstacle info
[jointTrajectory_obstacle, Trajectory_obstacle] = full_human_joints_rand();

%Determine XYZ position of each joint at each pose
a = 22;  %Number of points used to define human
[~,b] = size(jointTrajectory_obstacle);
points_obstacle = zeros(4,a,b);

%Right arm (use 1 in function)
for ii = 1:b
    points_obstacle(:,:,ii) = body(jointTrajectory_obstacle(:,ii),a);
end

%Create articulated obstacles to interact with the robot's sweep
start = 2;
human = generate_surface(points_obstacle,visual_sweep,Trajectory_obstacle,start,grid_spacing);
grid on
xlim([-1 1])
ylim([-2 0])
zlim([0 1])
xlabel('X')
ylabel('Y')
zlabel('Z')
hold on

%Construct a best fit for each joint

x = points_obstacle(1,:,5);
y = points_obstacle(2,:,5);
sliceNumber = points_obstacle(3,:,5);
coefficientsX = polyfit(sliceNumber, x, 1);
coefficientsY = polyfit(sliceNumber, y, 1);
sFitted = 1 : max(sliceNumber);
xFitted = polyval(coefficientsX, sFitted);
yFitted = polyval(coefficientsY, sFitted);
plot(sliceNumber, x, 'b*');
hold on;
plot(sliceNumber, y, 'c*');
plot(sFitted, xFitted, 'm-', 'LineWidth', 2);
plot(sFitted, yFitted, 'r-', 'LineWidth', 2);
grid on;
xlabel('Slice Number', 'FontSize', 20);
ylabel('x or y', 'FontSize', 20);
legend('x data', 'y data', 'x fit', 'yFit');
%Determine the curvature of the line of each joint
%use the most line with the most curvature to define the resolution