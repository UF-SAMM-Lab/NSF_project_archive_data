function [normal] = patch_normal(normal,P,ind)
    %The purpose of this function is to calculate the average normal value of
    %the current Coon's patch. The norma is calculated from three points
    mid_u = floor(length(P(:,1,1))/2);
    mid_v = floor(length(P(1,:,1))/2);
    P1 = [P(mid_u,mid_v,1), P(mid_u,mid_v,2), P(mid_u,mid_v,3)];
    P2 = [P(1,1,1), P(1,1,2), P(1,1,3)];
    P3 = [P(1,end,1), P(1,end,2), P(1,end,3)];
    norm = cross(P1-P2, P1-P3);
    norm = norm/sqrt(norm(1)^2+norm(2)^2+norm(3)^2);
    normal(1:3,ind) = norm(1:3);
end

