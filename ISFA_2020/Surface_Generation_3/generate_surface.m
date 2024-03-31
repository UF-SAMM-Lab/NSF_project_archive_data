function [patches] = generate_surface(points,visual,Trajectory,start,grid_spacing)
%--------------------------------------------------------------------------
%The purpose of this function is create coon's patch using a set of points.

%INPUT: points: XYZ locations of each joint at the selected poses

%OUTPUT: patches: a set of surfaces the compositely represend the entire
%area swept out by the manipulator during execution of the trajectory
%--------------------------------------------------------------------------

%Reshape into curves to feed into coons patches algorithm
[~, b, c] = size(points);

if start
    last_joint = start;
else
    last_joint = 2;
end
%Initialize vector to store patches
patches.x = [];
patches.y = [];
patches.z = [];
patches.time = [];
patches.segment = [];
patches.joint = [];

    for ii = 1:c-1 %Step through each region between poses
        for jj = b:-1:last_joint %Step through each region between joints
                      
            %Determine required mesh size according to size of the patch            
            [nbf_u,nbf_v] = mesh_setter(points, ii, jj, grid_spacing);
            
            %Define boundary curves for Coons Patch
            [c1, c2, c3, c4] = curves(nbf_u, nbf_v, points, ii, jj);
            
            %Create patches to describe the position thorughout the surface
            P = discrete_coons_patch(c1, c2, c3, c4, nbf_u, nbf_v);
                           
            %Create a patch to describe the time throughout the surface
            time = zeros(1,nbf_u);            
            time(1) = Trajectory(1,ii+1);
            time(nbf_u) = Trajectory(1,ii);
            time = time_filler(time,nbf_u);
            
            if visual == 1
                T = zeros(length(time),nbf_v);

                for p = 1:nbf_v
                    T(:,p) = time';
                end
                hold on 
                surf(P(:,:,1),P(:,:,2),P(:,:,3),T(:,:))
            end
            
            %Recast data into a stucture
            patches = patch_structure(P, nbf_v, nbf_u, time, Trajectory, jj, ii, patches);
            
        end
    end
end
