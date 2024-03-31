function visualize(occGrid,collision,visual_robot,visual_sweep,visual_obstacle,jointTrajectory,visual_object_out,obstacle,visual_time_compare,patches)

if visual_obstacle == 1
    scatter3(occGrid.x, occGrid.y, occGrid.z, 1, occGrid.time, 'o');
    hold on
end
if visual_sweep == 1       
    scatter3(collision(:,1),collision(:,2),collision(:,3),'filled','ro')
    grid on
    xlim([-1 1])
    ylim([-1 1])
    zlim([0 1])
    xlabel('X')
    ylabel('Y')
    zlabel('Z')
    hold on
end
if visual_robot == 1
    %Load edo from URDF and STL Meshes
    edo = importrobot('edo_sim.urdf');
    config = homeConfiguration(edo);
    [~,steps] = size(jointTrajectory);
    for a = 1:steps
        config(1).JointPosition = jointTrajectory(2,a);
        config(2).JointPosition = jointTrajectory(3,a);
        config(3).JointPosition = jointTrajectory(4,a);
        config(4).JointPosition = jointTrajectory(5,a);
        config(5).JointPosition = jointTrajectory(6,a);
        config(6).JointPosition = jointTrajectory(7,a);
        show(edo,config)
        hold on
    end
end
if visual_object_out
    if obstacle
        figure
        scatter3(obstacle.x,obstacle.y,obstacle.z)
        title('Collision Object')
        grid on
        xlim([-1 1])
        ylim([-1 1])
        zlim([0 1])
        xlabel('X')
        ylabel('Y')
        zlabel('Z')
    end
end
if visual_time_compare
    figure
    scatter3(occGrid.x,occGrid.y,occGrid.z,15,occGrid.time,'filled')
    hold on
    scatter3(patches.x,patches.y,patches.z,15,patches.time,'filled')
    xlabel('X')
    ylabel('Y')
    zlabel('Z')
    bar = colorbar;
    title(bar,'time (seconds)')
end
end

