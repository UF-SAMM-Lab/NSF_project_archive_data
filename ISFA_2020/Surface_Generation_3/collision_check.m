function [collision, segments, obstacle] = collision_check(data1, data2,timespan,visual_sweep,grid_spacing)
%--------------------------------------------------------------------------
%The purpose of this function is to compare the data between the surface
%generated from the robot trajectory and the cone of uncertainty from the
%obstacle

%The comparison is accomplished by descritizing the points in each data
%set, inputting them into cartesian grids, and comparing the grids
%--------------------------------------------------------------------------

%Convert data to a grid with a known spacing so that they are comparable
grid1 = Descritize_XYZ(grid_spacing, data1);  
grid2 = Descritize_XYZ(grid_spacing, data2);

%Descritize data time stamp into compare them more accurately
time_step = timespan/80;
time1 = Descritize_time(time_step, data1.time);
time2 = Descritize_time(time_step, data2.time);

%Recast into a form that would allow for quick lookup of collision info
cartesian_set1 = [grid1.x',grid1.y',grid1.z',time1'];
cartesian_set2 = [grid2.x',grid2.y',grid2.z',time2']; 

%A collision occurs any time the two sets have the same XYZ coordinate
%Determine all segments affected by the obstacle
[collision, index1, index2] = intersect(cartesian_set1,cartesian_set2,'rows');
segments = unique(data1.segment(index1));

if collision
    
    ind2 = find(cartesian_set2(:,4) == cartesian_set2(index2(1),4));
    if visual_sweep
        ind1 = find(cartesian_set1(:,4) == cartesian_set1(index1(1),4));
        %scatter3(cartesian_set1(ind1,1),cartesian_set1(ind1,2),cartesian_set1(ind1,3),'r.')
        %scatter3(cartesian_set2(ind2,1),cartesian_set2(ind2,2),cartesian_set2(ind2,3),'r.')
    end
end

%The purpose of the try catch is to differentiate between articulated
%obstacles and non articulated obstacles (cone of uncertainty)
try
    %Determine all object instances to report back to the replanner
    object = unique(data2.object(index2));
    object_index = ismember(data2.object,object);
    obstacle.x = data2.x(object_index);
    obstacle.y = data2.y(object_index);
    obstacle.z = data2.z(object_index);
catch
    if collision
        obstacle.x = cartesian_set2(ind2,1);
        obstacle.y = cartesian_set2(ind2,2);
        obstacle.z = cartesian_set2(ind2,3);
    else
        obstacle = 0;
    end
end

end

