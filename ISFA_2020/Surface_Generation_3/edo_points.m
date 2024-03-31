function [joint_points] = edo_points(joints, length)

%--------------------------------------------------------------------------
%The purpose of this function is to determine the location of the origin of
%each joint at the given step in time

%INPUT: joint_angles: an array that holds each joint angle
%Ouput: joint_points: an array that holds the location of each joint
%--------------------------------------------------------------------------

%Define transformations matricies and rotation matricies between joints. 
%These are the transformations as designed in the edo robot according to the URDF 
T1f = [1   0   0   0;
       0   1   0   0;
       0   0   1   0.135;
       0   0   0   1];

T21 = [1   0   0   0;
       0   1   0   0;
       0   0   1   0.202;
       0   0   0   1];
   
T32 = [1   0   0   0;
       0   1   0   0;
       0   0   1   0.2105;
       0   0   0   1];
   
T43 = [1   0   0   0;
       0   1   0   0;
       0   0   1   0.134;
       0   0   0   1];
   
T54 = [1   0   0   0;
       0   1   0   0;
       0   0   1   0.134;
       0   0   0   1];
  
T65 = [1   0   0   0;
       0   1   0   0;
       0   0   1   0.1745;
       0   0   0   1];

%Rotation matricies are fucntions of edo axis of rotation and joint angle rotation
m1 = [0 0 1];
R1 = rot_about_axis(m1,joints(2));
m2 = [1 0 0];
R2 = rot_about_axis(m2,joints(3));
m3 = [1 0 0];
R3 = rot_about_axis(m3,joints(4));
m4 = [0 0 1];
R4 = rot_about_axis(m4,joints(5));
m5 = [1 0 0];
R5 = rot_about_axis(m5,joints(6));

%Define joint points
joint_points  = zeros(4,length); 

%Position of the origin
joint_points(:,1) = [0 0 0 1]';

%Translation of joint one
A = T1f*R1;
joint_points(:,1) =  A*[0.07  0 0 1]';
joint_points(:,12) = A*[-0.07 0 0 1]';

%Translation of joint two after rotating about z
B = A*T21*R2;
joint_points(:,2) =  B*[0.14  0 0 1]';
joint_points(:,11) = B*[-0.14 0 0 1]';

%Translation of joint three after rotating about z
C = B*T32*R3;
joint_points(:,3) =  C*[0.14  0 0 1]';
joint_points(:,10) = C*[-0.14 0 0 1]';

%Translation of joint four after rotating about z
D = C*T43*R4;
joint_points(:,4) =  D*[0.1  0 0 1]';
joint_points(:,9) = D*[-0.1 0 0 1]';

%Translation of joint five after rotating about z
E = D*T54*R5;
joint_points(:,5) =  E*[0.07  0 0 1]';
joint_points(:,8) = E*[-0.07 0 0 1]';

%Translation of joint six after rotating about z (Rotation of end effector
%doesnt change any subsequent positions and thus doesnt need to be applied)
F = E*T65;
joint_points(:,6) = F*[0.07 0 0 1]';
joint_points(:,7) = F*[-0.07 0 0 1]';

end