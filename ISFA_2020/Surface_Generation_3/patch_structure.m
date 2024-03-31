function [patches] = patch_structure(P, nbf_v, nbf_u, time, Trajectory, jj, ii, patches)
            
    %Recast data into a stucture
    Px = reshape(P(:,:,1),1,[]);
    Py = reshape(P(:,:,2),1,[]);
    Pz = reshape(P(:,:,3),1,[]);            
    patches.x = [patches.x Px];
    patches.y = [patches.y Py];
    patches.z = [patches.z Pz];
    for kk = 1:nbf_v
        patches.time = [patches.time time];
        patches.segment = [patches.segment ones(1,nbf_u)*Trajectory(2,ii)];
        patches.joint = [patches.joint jj*ones(1,nbf_u)];
    end

end

