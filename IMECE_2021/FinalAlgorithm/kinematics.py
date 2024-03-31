#!/usr/bin/env python
import numpy as np
import math

def rotAboutAxis(m,theta):
    #The purpose of this function is to take in an arbitrary axis and angle (rad)
    #of rotation and output the rotation matrix
    
    #Definintion of axis, sin, and cos of angle, and composite term v
    mx = m[0]
    my = m[1]
    mz = m[2]
    s = math.sin(theta)
    c = math.cos(theta)
    v = 1 - c
    
    R = np.array([[mx*mx*v+c,      mx*my*v-mz*s,      mx*mz*v+my*s],
                  [mx*my*v+mz*s,   my*my*v+c,         my*mz*v-mx*s],
                  [mx*mz*v-my*s,   my*mz*v+mx*s,      mz*mz*v+c   ]])
    
    #Put in transformation matrix
    Tout = np.zeros(shape=(4,4))
    Tout[3,3] = 1
    Tout[0:3,0:3] = R[0:3,0:3]
    
    return(Tout)

def transMatricesCloud():
    T1f = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    T21 = T1f
    T32 = T1f
    T43 = T1f
    T54 = T1f
    
    return T1f, T21, T32, T43, T54

def transMatrices():
    #Define transformations matricies and rotation matricies between joints. 
    #These are the transformations as designed in the edo robot according to the URDF 
    T1f = np.array([[1, 0, 0, -0.3],
                    [0, 1, 0, -1],
                    [0, 0, 1, 0.75],
                    [0, 0, 0, 1]])

    T21 = np.array([[1, 0,  0, 0],
                    [0, 1,  0, 0],
                    [0, 0,  1, 0.202],
                    [0, 0,  0, 1]])

    T32 = np.array([[1,  0,  0, 0],
                    [0,  1,  0, 0.0],
                    [0,  0,  1, 0.2105],
                    [0,  0,  0, 1]])

    T43 = np.array([[1,  0,  0,  0],
                    [0,  1,  0,  0],
                    [0,  0,  1,  0.134],
                    [0,  0,  0,  1]])

    T54 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.134],
                    [0, 0, 0, 1]])

    T65 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.1745],
                    [0, 0, 0, 1]])
    
    return T1f, T21, T32, T43, T54, T65

def rotMatrices(joints):
    #Rotation matricies are functions of edo axis of rotation and joint angle rotation
    m1 = [0, 0, 1]
    R1 = rotAboutAxis(m1,joints[1])
    m2 = [1, 0, 0]
    R2 = rotAboutAxis(m2,joints[2])
    m3 = [1, 0, 0]
    R3 = rotAboutAxis(m3,joints[3])
    m4 = [0, 0, 1]
    R4 = rotAboutAxis(m4,joints[4])
    m5 = [1, 0, 0]
    R5 = rotAboutAxis(m5,joints[5])

    return R1, R2, R3, R4, R5


def JointPoints_shift(joints, length):
   
    #--------------------------------------------------------------------------
    #The purpose of this function is to determine the location of the origin of
    #each joint at the given step in time
    
    #INPUT: joint_angles: an array that holds each joint angle
    #Ouput: joint_points: an array that holds the location of each joint
    #--------------------------------------------------------------------------
    
    #Define transformations matricies and rotation matricies between joints. 
    #These are the transformations as designed in the edo robot according to the URDF 
    T1f, T21, T32, T43, T54, T65 = transMatrices()
    
    #Rotation matricies are functions of edo axis of rotation and joint angle rotation
    R1, R2, R3, R4, R5 = rotMatrices(joints)
    
    #Define joint points
    joint_points  = np.zeros(shape=(4,length))
    
    #Translation of joint one
    A = np.matmul(T1f, R1)
    joint_points[:,0]  = np.matmul(A, np.transpose([0,  0.07, 0, 1]))
    joint_points[:,11] = np.matmul(A, np.transpose([0, -0.07, 0, 1]))
    
    #Translation of joint two after rotating 
    B = np.matmul(A, np.matmul(T21, R2))
    joint_points[:,1] = np.matmul(B ,np.transpose([0,   0.12, 0, 1]))
    joint_points[:,10] = np.matmul(B ,np.transpose([0, -0.12, 0, 1]))
    
    #Translation of joint three after rotating
    C = np.matmul(B, np.matmul(T32, R3))
    joint_points[:,2] = np.matmul(C, np.transpose([0,  0.12, 0, 1]))
    joint_points[:,9] = np.matmul(C, np.transpose([0, -0.12, 0, 1]))
    
    #Translation of joint four after rotating 
    D = np.matmul(C, np.matmul(T43, R4))
    joint_points[:,3] = np.matmul(D, np.transpose([0,  0.1, 0, 1]))
    joint_points[:,8] = np.matmul(D, np.transpose([0, -0.1, 0,  1]))
    
    #Translation of joint five after rotating 
    E = np.matmul(D, np.matmul(T54, R5))
    joint_points[:,4] = np.matmul(E, np.transpose([0, 0.07, 0, 1]))
    joint_points[:,7] = np.matmul(E, np.transpose([0, -0.07, 0, 1]))
    
    #Translation of joint six
    F = np.matmul(E, T65)
    joint_points[:,5] = np.matmul(F, np.transpose([0,  0.07, 0, 1]))
    joint_points[:,6] = np.matmul(F, np.transpose([0, -0.07, 0, 1]))
    
    return joint_points

def JointPoints(joints, length):
   
    #--------------------------------------------------------------------------
    #The purpose of this function is to determine the location of the origin of
    #each joint at the given step in time
    
    #INPUT: joint_angles: an array that holds each joint angle
    #Ouput: joint_points: an array that holds the location of each joint
    #--------------------------------------------------------------------------
    
    #Define transformations matricies and rotation matricies between joints. 
    #These are the transformations as designed in the edo robot according to the URDF 
    T1f, T21, T32, T43, T54, T65 = transMatrices()
    
    #Rotation matricies are functions of edo axis of rotation and joint angle rotation
    R1, R2, R3, R4, R5 = rotMatrices(joints)

    #Define joint points
    joint_points  = np.zeros(shape=(4,length))
    
    #Translation of joint one
    A = np.matmul(T1f, R1)
    joint_points[:,0] = np.matmul(A, np.transpose([0.07, 0, 0, 1]))
    joint_points[:,11] = np.matmul(A, np.transpose([-0.07, 0, 0, 1]))
    
    #Translation of joint two after rotating 
    B = np.matmul(A, np.matmul(T21, R2))
    joint_points[:,1] = np.matmul(B ,np.transpose([  0.12, 0, 0, 1]))
    joint_points[:,10] = np.matmul(B ,np.transpose([-0.12, 0, 0, 1]))
    
    #Translation of joint three after rotating
    C = np.matmul(B, np.matmul(T32, R3))
    joint_points[:,2] = np.matmul(C, np.transpose([ 0.12, 0, 0, 1]))
    joint_points[:,9] = np.matmul(C, np.transpose([-0.12, 0, 0, 1]))
    
    #Translation of joint four after rotating 
    D = np.matmul(C, np.matmul(T43, R4))
    joint_points[:,3] = np.matmul(D, np.transpose([ 0.1, 0, 0, 1]))
    joint_points[:,8] = np.matmul(D, np.transpose([-0.1, 0, 0, 1]))
    
    #Translation of joint five after rotating 
    E = np.matmul(D, np.matmul(T54, R5))
    joint_points[:,4] = np.matmul(E, np.transpose([ 0.07, 0, 0, 1]))
    joint_points[:,7] = np.matmul(E, np.transpose([-0.07, 0, 0, 1]))
    
    #Translation of joint six
    F = np.matmul(E, T65)
    joint_points[:,5] = np.matmul(F, np.transpose([ 0.07, 0, 0, 1]))
    joint_points[:,6] = np.matmul(F, np.transpose([-0.07, 0, 0, 1]))
    
    return joint_points

def pointCloudShift(cloud,joints):
   
    #Define transformations matricies and rotation matricies between joints. 
    #These are the transformations as designed in the edo robot according to the URDF 
    T1f, T21, T32, T43, T54 = transMatricesCloud()

    #Rotation matricies are functions of edo axis of rotation and joint angle rotation
    R1, R2, R3, R4, R5 = rotMatrices(joints)
    
    #Translation of points on link one
    A = np.matmul(T1f, R1)
    old = 0
    numPoints = 80
    new = old + numPoints
    temp_cloud = np.ones((4,numPoints))
    temp_cloud[0:3,:] = cloud[0:3,old:new]
    temp_cloud[0:4,:] = np.matmul(A, temp_cloud[0:4,:])
    cloud[0:3,old:new] = temp_cloud[0:3,:]

    #Translation of points on link two
    B = np.matmul(A, np.matmul(T21, R2))
    numPoints = 90
    old = new
    new = old + numPoints
    temp_cloud = np.ones((4,numPoints))
    temp_cloud[0:3,:] = cloud[0:3,old:new]
    temp_cloud[0:4,:] = np.matmul(B, temp_cloud[0:4,:])
    cloud[0:3,old:new] = temp_cloud[0:3,:]
    
    #Translation of points on link three
    C = np.matmul(B, np.matmul(T32, R3))
    numPoints = 48
    old = new
    new = old + numPoints
    temp_cloud = np.ones((4,numPoints))
    temp_cloud[0:3,:] = cloud[0:3,old:new]
    temp_cloud[0:4,:] = np.matmul(C, temp_cloud[0:4,:])
    cloud[0:3,old:new] = temp_cloud[0:3,:]
    
    #Translation of points on link four 
    D = np.matmul(C, np.matmul(T43, R4))
    numPoints = 36
    old = new
    new = old + numPoints
    temp_cloud = np.ones((4,numPoints))
    temp_cloud[0:3,:] = cloud[0:3,old:new]
    temp_cloud[0:4,:] = np.matmul(D, temp_cloud[0:4,:])
    cloud[0:3,old:new] = temp_cloud[0:3,:]    

    #Translation of points on link five
    E = np.matmul(D, np.matmul(T54, R5))
    numPoints = 32
    old = new
    new = old + numPoints
    temp_cloud = np.ones((4,numPoints))
    temp_cloud[0:3,:] = cloud[0:3,old:new]
    temp_cloud[0:4,:] = np.matmul(E, temp_cloud[0:4,:])
    cloud[0:3,old:new] = temp_cloud[0:3,:]

    return cloud

