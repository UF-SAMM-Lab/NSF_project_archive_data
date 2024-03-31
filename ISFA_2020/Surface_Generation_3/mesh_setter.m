function [nbf_u,nbf_v] = mesh_setter(points, ii, jj, grid_spacing)
%The purpose of this function is to evaluate the size of the patch to be
%created by the four points and set the mesh so that its spacing is at most half the
%size of the overall grid spacing.

%Evaluate the XYZ Distance between each point
line1 = sqrt((points(1,jj,ii+1)-points(1,jj,ii))^2+(points(2,jj,ii+1)-points(2,jj,ii))^2+(points(3,jj,ii+1)-points(3,jj,ii))^2);
line2 = sqrt((points(1,jj-1,ii+1)-points(1,jj,ii+1))^2+(points(2,jj-1,ii+1)-points(2,jj,ii+1))^2+(points(3,jj-1,ii+1)-points(3,jj,ii+1))^2);
line3 = sqrt((points(1,jj-1,ii+1)-points(1,jj-1,ii))^2+(points(2,jj-1,ii+1)-points(2,jj-1,ii))^2+(points(3,jj-1,ii+1)-points(3,jj-1,ii))^2);
line4 = sqrt((points(1,jj,ii)-points(1,jj-1,ii))^2+(points(2,jj,ii)-points(2,jj-1,ii))^2+(points(3,jj,ii)-points(3,jj-1,ii))^2);

%Use the longest boundary curve to define the mesh
longest_u = max([line1,line3]);
nbf_u = ceil(longest_u/(grid_spacing/2));
longest_v = max([line2,line4]);
nbf_v = ceil(longest_v/(grid_spacing/2));

%In order to establish a normal, the mesh must be greated than 1 x 1
if nbf_u <= 1
    nbf_u = 2;
end

if nbf_v <= 1
    nbf_v = 2;
end

end

