function Coons_patch = discrete_coons_patch(c1, c2, c3, c4, nbf_u, nbf_v)
% discrete_coons_patch : function to compute the control points
% of the Coons patch of a given contour
% defined by four curves, c1, c2, c3, and c4.

s1 = size(c1);
s2 = size(c2);
s3 = size(c3);
s4 = size(c4);

%% Contour initialization
Coons_patch = zeros(nbf_u,nbf_v,3);
% c1_spl_vect = zeros(nbf_u,1);
% c1_spl_vect(1,1) = 1;
% c1_spl_vect(end,1) = s1(1);
% c1_spl_vect(2:end-1,1) = round((s1(1)) * round(100*(2:nbf_u-1)/(nbf_u-1))/100);
% c3_spl_vect = zeros(nbf_u,1);
% c3_spl_vect(1,1) = 1;
% c3_spl_vect(end,1) = s3(1);
% c3_spl_vect(2:end-1,1) = round((s3(1)) * round(100*(1:nbf_u-2)/(nbf_u-1))/100);
% c2_spl_vect = zeros(nbf_v,1);
% c2_spl_vect(1,1) = 1;
% c2_spl_vect(end,1) = s2(1);
% c2_spl_vect(2:end-1,1) = round((s2(1)) * round(100*(1:nbf_v-2)/(nbf_v-1))/100);
% c4_spl_vect = zeros(nbf_v,1);
% c4_spl_vect(1,1) = 1;
% c4_spl_vect(end,1) = s4(1);
% c4_spl_vect(2:end-1,1) = round((s4(1)) * round(100*(1:nbf_v-2)/(nbf_v-1))/100);

%Alternative Approach (replaces lines 13-28)
c1_spl_vect = zeros(nbf_u,1);
c1_spl_vect(1,1) = 1;
c1_spl_vect(end,1) = s1(1);
c1_spl_vect(2:end-1,1) = 2:nbf_u-1;
c3_spl_vect = zeros(nbf_u,1);
c3_spl_vect(1,1) = 1;
c3_spl_vect(end,1) = s3(1);
c3_spl_vect(2:end-1,1) = 2:nbf_u-1;
c2_spl_vect = zeros(nbf_v,1);
c2_spl_vect(1,1) = 1;
c2_spl_vect(end,1) = s2(1);
c2_spl_vect(2:end-1,1) = 2:nbf_v-1;
c4_spl_vect = zeros(nbf_v,1);
c4_spl_vect(1,1) = 1;
c4_spl_vect(end,1) = s4(1);
c4_spl_vect(2:end-1,1) = 2:nbf_v-1;

Coons_patch(end:-1:1,1,:)   = c1(c1_spl_vect,1,:);
Coons_patch(end:-1:1,end,:) = c3(c3_spl_vect,1,:);
Coons_patch(1,:,:)   = c2(c2_spl_vect,1,:);
Coons_patch(end,:,:) = c4(c4_spl_vect,1,:);
%% Patch values computation
for n = 1:nbf_u
    
    for p = 1:nbf_v
        u = (n-1)/(nbf_u-1);
        v = (p-1)/(nbf_v-1);
        Coons_patch(n,p,1) = (1-u)*Coons_patch(1,p,1)+...
                             (1-v)*Coons_patch(n,1,1)+...
                                 u*Coons_patch(end,p,1) +...
                                 v*Coons_patch(n,end,1) +...
                                 ...
                       (u-1)*(1-v)*Coons_patch(1,1,1)-...
                               u*v*Coons_patch(end,end,1)+...
                           u*(v-1)*Coons_patch(end,1,1)+...
                           v*(u-1)*Coons_patch(1,end,1);
        Coons_patch(n,p,2) = (1-u)*Coons_patch(1,p,2)+...
                             (1-v)*Coons_patch(n,1,2)+...
                                 u*Coons_patch(end,p,2) +...
                                 v*Coons_patch(n,end,2) +...
                                 ...
                       (u-1)*(1-v)*Coons_patch(1,1,2)-...l.pl
                               u*v*Coons_patch(end,end,2)+...
                           u*(v-1)*Coons_patch(end,1,2)+...
                           v*(u-1)*Coons_patch(1,end,2);
        Coons_patch(n,p,3) = (1-u)*Coons_patch(1,p,3)+...
                             (1-v)*Coons_patch(n,1,3)+...
                                 u*Coons_patch(end,p,3) +...
                                 v*Coons_patch(n,end,3) +...
                                 ...
                       (u-1)*(1-v)*Coons_patch(1,1,3)-...
                               u*v*Coons_patch(end,end,3)+...
                           u*(v-1)*Coons_patch(end,1,3)+...
                           v*(u-1)*Coons_patch(1,end,3);
    end
    
end
end % discrete_coons_patch