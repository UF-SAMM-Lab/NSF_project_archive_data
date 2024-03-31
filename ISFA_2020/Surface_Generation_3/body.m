function [joint_points] = body(joints,length)

%--------------------------------------------------------------------------
%The purpose of this function is to determine the location of the origin of
%each joint at the given step in time

%INPUT: joint_angles: an array that holds each joint angle
%Ouput: joint_points: an array that holds the location of each joint
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%Safety factors
%--------------------------------------------------------------------------
head = 1; neck = 1;
shoulderR = 1; armR = 1; forearmR = 1;
shoulderL = 1; armL = 1; forearmL = 1;
waist = 1; torso = 1;

%--------------------------------------------------------------------------
%Define transformations matricies and rotation matricies between joints. 
%--------------------------------------------------------------------------
T1f = [1   0   0   -0.1;
       0   1   0   -0.8;
       0   0   1   0.1;
       0   0   0   1];

T21 = [1   0   0   0.23;
       0   1   0   0;
       0   0   1   0.4;
       0   0   0   1];

T32 = [1   0   0   0;
       0   1   0   0;
       0   0   1   -0.2;
       0   0   0   1];

T43 = [1   0   0   0;
       0   1   0   0;
       0   0   1   -0.2;
       0   0   0   1];

T51 = [1   0   0   -0.23;
       0   1   0   0;
       0   0   1   0.4;
       0   0   0   1];

T65 = [1   0   0   0;
       0   1   0   0;
       0   0   1   -0.2;
       0   0   0   1];

T76 = [1   0   0   0;
       0   1   0   0;
       0   0   1   -0.2;
       0   0   0   1];

T81 = [1   0   0   0;
       0   1   0   0;
       0   0   1   0.45;
       0   0   0   1];
   
%--------------------------------------------------------------------------
%Rotation matricies are fucntions of axis of rotation and joint angle rotation
%Define rotations about each joint
%--------------------------------------------------------------------------

%Torso
m1a = [1 0 0];
R1a = rot_about_axis(m1a,joints(1));
m1b = [0 1 0];
R1b = rot_about_axis(m1b,joints(2));
m1c = [0 0 1];
R1c = rot_about_axis(m1c,joints(3));

%Right shoulder
m2a = [1 0 0];
R2a = rot_about_axis(m2a,joints(4));
m2b = [0 1 0];
R2b = rot_about_axis(m2b,joints(5));
m2c = [0 0 1];
R2c = rot_about_axis(m2c,joints(6));

%Right elbow
m3 = [1 0 0];
R3 = rot_about_axis(m3,joints(7));

%Left shoulder
m5a = [1 0 0];
R5a = rot_about_axis(m5a,joints(8));
m5b = [0 1 0];
R5b = rot_about_axis(m5b,joints(9));
m5c = [0 0 1];
R5c = rot_about_axis(m5c,joints(10));

%Left elbow
m6 = [1 0 0];
R6 = rot_about_axis(m6,joints(11));

%Neck shoulder
m8a = [1 0 0];
R8a = rot_about_axis(m8a,joints(12));
m8b = [0 1 0];
R8b = rot_about_axis(m8b,joints(13));
m8c = [0 0 1];
R8c = rot_about_axis(m8c,joints(14));

%--------------------------------------------------------------------------
%Use transformation and rotation matricies to perform forward kinematics to
%determine the cartesian location of each joint and determine boundary points. 
%Apply to safety factors to expand the boundaries around the joints.
%--------------------------------------------------------------------------

%Define joint points
joint_points  = zeros(4,length);

%Translation of joint one
A = T1f*R1a*R1b*R1c;
%Shoulder (right) points with safety factor
sf = waist;
joint_points(:,1)  = A*[ 0.12*sf 0 0 1]';
joint_points(:,22) = A*[-0.12*sf 0 0 1]';

%Translation of joint two
B = A*T21*R2a*R2b*R2c;
%Shoulder (right) points with safety factor
sf = max([shoulderR torso]); %Use the larger of the two safety factors at the junction
joint_points(:,2) = B*[-0.03    0 -0.07*sf 1]';
joint_points(:,7) = B*[ 0.03*sf 0  0.04*sf 1]';

%Translation of joint three
C = B*T32*R3;
%Arm (right) points with safety factor
sf = max([forearmR armR]); %Use the larger of the two safety factors at the junction
joint_points(:,3) = C*[-0.03*sf 0 0 1]';
joint_points(:,6) = C*[ 0.03*sf 0 0 1]';

%Translation of joint four
D = C*T43;
%Forearm (right) points with safety factor
sf = forearmR;
joint_points(:,4) = D*[-0.03*sf 0 0 1]';
joint_points(:,5) = D*[ 0.03*sf 0 0 1]';

%Translation of joint five
E = A*T51*R5a*R5b*R5c;
%Shoulder (left) points with safety factor
sf = max([shoulderL torso]); %Use the larger of the two safety factors at the junction
joint_points(:,21) = E*[ 0.03    0 -0.07*sf 1]';
joint_points(:,16) = E*[-0.03*sf 0  0.04*sf 1]';

%Translation of joint six
F = E*T65*R6;
%Arm (left) points with safety factor
sf = max([forearmL armL]); %Use the larger of the two safety factors at the junction
joint_points(:,17) = F*[-0.03*sf 0 0 1]';
joint_points(:,20) = F*[ 0.03*sf 0 0 1]';


%Translation of joint seven
G = F*T76;
%Forearm (left) points with safety factor
sf = forearmL;
joint_points(:,18) = G*[-0.03*sf 0 0 1]';
joint_points(:,19) = G*[ 0.03*sf 0 0 1]';


%Translation of joint eight
H = A*T81*R8a*R8b*R8c;
%Neck points with safety factor
sf = neck;
joint_points(:,8)  =   H*[ 0.035*sf 0  0        1]';
joint_points(:,15) =   H*[-0.035*sf 0  0        1]';
joint_points(:,9)  =   H*[ 0.035*sf 0  0.08     1]';
joint_points(:,14) =   H*[-0.035*sf 0  0.08     1]';
%Head points with safety factor
sf = head;
joint_points(:,10) =   H*[ 0.07*sf  0  0.08              1]';
joint_points(:,13) =   H*[-0.07*sf  0  0.08              1]';
joint_points(:,11) =   H*[ 0.07*sf  0  0.28+0.07*(sf-1)  1]';
joint_points(:,12) =   H*[-0.07*sf  0  0.28+0.07*(sf-1)  1]';




end