function [Tout] = rot_about_axis(m,theta)
%The purpose of this function is to take in an arbitrary axis and angle (rad)
%of rotation and output the rotation matrix

%Definintion of axis, sin, and cos of angle, and composite term v
mx = m(1);
my = m(2);
mz = m(3);
s = sin(theta);
c = cos(theta);
v = 1- c;

R = [mx*mx*v+c      mx*my*v-mz*s      mx*mz*v+my*s;
     mx*my*v+mz*s   my*my*v+c         my*mz*v-mx*s;
     mx*mz*v-my*s   my*mz*v+mx*s      mz*mz*v+c  ];

%Put in transformation matrix
Tout = zeros(4);
Tout(4,4) = 1;
Tout(1:3,1:3) = R(1:3,1:3);

end

