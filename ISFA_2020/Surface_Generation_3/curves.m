function [c1, c2, c3, c4] = curves(nbf_u, nbf_v, points, ii, jj)
            
    c1 = zeros(nbf_u,1,3);
    c2 = zeros(nbf_v,1,3);
    c3 = zeros(nbf_u,1,3);
    c4 = zeros(nbf_v,1,3);

    %Curve 1
    c1(1,1,1) = points(1,jj,ii)';
    c1(1,1,2) = points(2,jj,ii)';
    c1(1,1,3) = points(3,jj,ii)';
    c1(nbf_u,1,1) = points(1,jj,ii+1)';
    c1(nbf_u,1,2) = points(2,jj,ii+1)';
    c1(nbf_u,1,3) = points(3,jj,ii+1)';
    c1 = point_filler(c1,nbf_u);

    %Curve 3
    c3(1,1,1) = points(1,jj-1,ii)';
    c3(1,1,2) = points(2,jj-1,ii)';
    c3(1,1,3) = points(3,jj-1,ii)';
    c3(nbf_u,1,1) = points(1,jj-1,ii+1)';
    c3(nbf_u,1,2) = points(2,jj-1,ii+1)';
    c3(nbf_u,1,3) = points(3,jj-1,ii+1)';
    c3 = point_filler(c3,nbf_u);

    %Curve 2
    c2(1,1,1) = c1(nbf_u,1,1);
    c2(1,1,2) = c1(nbf_u,1,2);
    c2(1,1,3) = c1(nbf_u,1,3);
    c2(nbf_v,1,1) = c3(nbf_u,1,1);
    c2(nbf_v,1,2) = c3(nbf_u,1,2);
    c2(nbf_v,1,3) = c3(nbf_u,1,3);
    c2 = point_filler(c2,nbf_v);

    %Curve 4
    c4(1,1,1) = c1(1,1,1);
    c4(1,1,2) = c1(1,1,2);
    c4(1,1,3) = c1(1,1,3);
    c4(nbf_v,1,1) = c3(1,1,1);
    c4(nbf_v,1,2) = c3(1,1,2);
    c4(nbf_v,1,3) = c3(1,1,3);
    c4 = point_filler(c4,nbf_v);
   
end

