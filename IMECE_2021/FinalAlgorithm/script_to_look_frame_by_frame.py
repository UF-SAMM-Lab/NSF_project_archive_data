#!/usr/bin/env my_pymc_env
from cv2 import data
import util as cmr
import cv2
import time
import pyrealsense2 as rs
import numpy as np
import pickle
from skeletontracker import skeletontracker
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from testing_data import *
from kinematics import *
from generate_surface import *
from collision_check import collisionCheck
from collections import defaultdict
from spatialmath.base import * #Peter Corke Spatial Math Toolbox
from scipy.signal import butter, lfilter

def kalmanFilter(points):
    global x_hat
    global x_hat_next
    global actuals
    global dims
    global init_kalman
    global a
    global b
    #global dims

    ang_success,theta = JointCarts2Angles(points) #get current human joint angles

    #Condition for first run
    if init_kalman:
        dims = np.array([np.linalg.norm(points[:,1]-points[:,0]),np.linalg.norm(points[:,2]-points[:,1]),np.linalg.norm(points[:,6]-points[:,3]),np.linalg.norm(points[:,4]-points[:,3]),np.linalg.norm(points[:,5]-points[:,4]),np.linalg.norm(points[:,7]-points[:,6]),np.linalg.norm(points[:,8]-points[:,7])])
        actuals = theta
        x_hat[:21] = theta
        x_hat_next[:21] = theta
    else:
        actuals = np.append(actuals,theta,axis=1) #store actual thetas

    init_kalman = False

    tmp = lfilter(b,a,actuals[:,-10:]) #smooth the computed angles with the low pass filter
    theta = tmp[:,-1].reshape((21,1)) #ensure angle array is correct shape
    zero_filter_err = False #boolean indicating the filter should be in "use predictions" mode
    if not ang_success: #if joint angle computation is invalid
        theta = x_hat_next[:21] #set the thetas to the predicted thetas from the filter
        zero_filter_err = True
    if zero_filter_err:
        x_hat[:21]=theta #set the filter theta states to the predicted values to prevent expontial divergence of filter states
    else: #only update the skeleton dimensions when the joint angle computation was valid
        dims = np.array([np.linalg.norm(points[:,1]-points[:,0]),np.linalg.norm(points[:,2]-points[:,1]),np.linalg.norm(points[:,6]-points[:,3]),np.linalg.norm(points[:,4]-points[:,3]),np.linalg.norm(points[:,5]-points[:,4]),np.linalg.norm(points[:,7]-points[:,6]),np.linalg.norm(points[:,8]-points[:,7])])
    x_hat, x_hat_next = kalman_filter(theta,x_hat) #get the est. and pred. states from the filter

    predicted_points = forwardKinematics(x_hat_next[:21],points[:,0],dims)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(points[0,:], points[1,:], points[2,:], color='blue', marker='.')  
    ax.scatter(predicted_points[0,:], predicted_points[1,:], predicted_points[2,:],color='red', marker='.')    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.axes.set_xlim3d(left=-1, right=1) 
    ax.axes.set_zlim3d(bottom=0, top=2) 
    ax.axes.set_ylim3d(bottom=-1, top=1) 
    plt.show()

    return predicted_points

def mapSkelToKalman(skelPoints):

    kalPoints = np.zeros((3,15))
    kalPoints[:,0] = (skelPoints[8,:] + skelPoints[11,:])/2
    kalPoints[:,1] = skelPoints[1,:]
    kalPoints[:,2] = skelPoints[0,:]
    kalPoints[:,3] = skelPoints[5,:]
    kalPoints[:,4] = skelPoints[6,:]
    kalPoints[:,5] = skelPoints[7,:]
    kalPoints[:,6] = skelPoints[2,:]
    kalPoints[:,7] = skelPoints[3,:]
    kalPoints[:,8] = skelPoints[4,:]
    kalPoints[:,9] = skelPoints[11,:]
    kalPoints[:,10] = skelPoints[12,:]
    kalPoints[:,11] = skelPoints[13,:]
    kalPoints[:,12] = skelPoints[8,:]
    kalPoints[:,13] = skelPoints[9,:]
    kalPoints[:,14] =  skelPoints[10,:]
    

    return kalPoints

def mapKalmanToSkel(kalPoints, skelPoints):

    skelPoints[0,:] = kalPoints[:,2]
    skelPoints[1,:] = kalPoints[:,1]
    skelPoints[2,:] = kalPoints[:,6]
    skelPoints[3,:] = kalPoints[:,7]
    skelPoints[4,:] = kalPoints[:,8]
    skelPoints[5,:] = kalPoints[:,3]
    skelPoints[6,:] = kalPoints[:,4]
    skelPoints[7,:] = kalPoints[:,5]

    return skelPoints

def JointCarts2Angles(points):
    #This function extracts the human joint angles from the 3x15 matrix of points, each
    #column is x,y,z for a skeleton point
    #Each joint has a roll,pitch,yaw for improved accuracy even though many joints
    #dont have 3 DOFs
    vects = np.zeros((3,14))
    vects[:,0] = points[:,1]-points[:,0] #pelvis->spine
    vects[:,1] = points[:,2]-points[:,1] #spine->head top
    vects[:,2] = points[:,3]-points[:,1] #L shoulder->spine
    vects[:,3] = points[:,6]-points[:,1] #L elbow->L shoulder
    vects[:,4] = points[:,4]-points[:,3] #L wrist->L elbow
    vects[:,5] = points[:,5]-points[:,4] #R shoulder->spine
    vects[:,6] = points[:,7]-points[:,6] #R elbow->R shoulder
    vects[:,7] = points[:,8]-points[:,7] #R wrist->R elbow
    vects[:,8] = points[:,9]-points[:,0] #L hip->pelvis
    vects[:,9] = points[:,12]-points[:,0] #L knee->L hip
    vects[:,10] = points[:,10]-points[:,9] #L ankle->L knee
    vects[:,11] = points[:,11]-points[:,10] #R hip->pelvis
    vects[:,12] = points[:,13]-points[:,12] #R knee->R hip
    vects[:,13] = points[:,14]-points[:,13] #R ankle->R knee
    shoulder2shoulder = points[:,6]-points[:,3] #L shoulder->R shoulder vector
    
    theta = np.zeros((21,1))
    success = True
    #go from fixed z axis to the pelvis->spine vector
    #align x axis with new z (pelvis->spine vector) crossed with fixed y
    norm_z1 = np.linalg.norm(vects[:,0])
    if norm_z1==0:
        success = False
        return success,theta
    z_1 = vects[:,0]/norm_z1
    x_1 = np.cross(np.array([0,1,0]).T,z_1)
    norm_x1 = np.linalg.norm(x_1)
    if norm_x1==0:
        success = False
        return success,theta
    x_1 = x_1/norm_x1
    y_1 = np.cross(z_1,x_1)
    norm_y1 = np.linalg.norm(y_1)
    if norm_y1==0:
        success = False
        return success,theta
    y_1 = y_1/norm_y1
    R1 = np.concatenate((x_1.reshape((3,1)),y_1.reshape((3,1)),z_1.reshape((3,1))),axis=1)
    theta[0], theta[1], theta[2] = tr2rpy(R1)
    
    #go from spine z axis (z1) to spine->head vector
    #new y axis aligned with new z (spine->head vector) crossed with spine x axis (x1)
    norm_z2 = np.linalg.norm(vects[:,1])
    if norm_z2==0:
        success = False
        return success,theta
    z_2 = vects[:,1]/norm_z2
    y_2 = np.cross(z_2,x_1)
    norm_y2 = np.linalg.norm(y_2)
    if norm_y2==0:
        success = False
        return success,theta
    y_2 = y_2/norm_y2
    x_2 = np.cross(y_2,z_2)
    norm_x2 = np.linalg.norm(x_2)
    if norm_x2==0:
        success = False
        return success,theta
    x_2 = x_2/norm_x2
    R2 = R1.T@np.concatenate((x_2.reshape((3,1)),y_2.reshape((3,1)),z_2.reshape((3,1))),axis=1)
    theta[3], theta[4], theta[5] = tr2rpy(R2)
    
    #go form spine x axis (x_1) to the shoulder->shoulder vector
    #new y axis is spine z axis (z_1) crossed with shoulder->shoulder vector
    norm_x3 = np.linalg.norm(shoulder2shoulder)
    if norm_x3==0:
        success = False
        return success,theta
    x_3 = shoulder2shoulder/norm_x3
    y_3 = np.cross(z_1,x_3)
    norm_y3 = np.linalg.norm(y_3)
    if norm_y3==0:
        success = False
        return success,theta
    y_3 = y_3/norm_y3
    z_3 = np.cross(x_3,y_3)
    norm_z3 = np.linalg.norm(z_3)
    if norm_z3==0:
        success = False
        return success,theta
    z_3 = z_3/norm_z3
    R3 = R1.T@np.concatenate((x_3.reshape((3,1)),y_3.reshape((3,1)),z_3.reshape((3,1))),axis=1)
    theta[6], theta[7], theta[8] = tr2rpy(R3)
   
    R_shoulder = R1@R3 #rotation matrix to go from fixed xyz at pelvis to shoulder xyz
    
    #go from shoulder z axis to the opposite of the shoulder1->elbow1 vector
    #align the elbox y axis with the shoulder->shoulder vector crossed with elbow1->shoulder1
    y_4 = np.cross(shoulder2shoulder,-vects[:,4])
    norm_y4 = np.linalg.norm(y_4)
    if norm_y4==0:
        success = False
        return success,theta
    y_4 = y_4/norm_y4
    z_4 = -vects[:,4]
    norm_z4 = np.linalg.norm(z_4)
    if norm_z4==0:
        success = False
        return success,theta
    z_4 = z_4/norm_z4
    x_4 = np.cross(y_4,z_4)
    norm_x4 = np.linalg.norm(x_4)
    if norm_x4==0:
        success = False
        return success,theta
    x_4 = x_4/norm_x4
    R_e1 = R_shoulder.T@np.concatenate((x_4.reshape((3,1)),y_4.reshape((3,1)),z_4.reshape((3,1))),axis=1)
    theta[9], theta[10], theta[11] = tr2rpy(R_e1)
    
    #align the elbow z axis with the elbow1->wrist1 vector
    #align the wrist y axis with the cross product of the elbow1->wrist1 vector and the shoulder1->elbow1 vector
    y_5 = np.cross(vects[:,5],vects[:,4])
    norm_y5 = np.linalg.norm(y_5)
    if norm_y5==0:
        success = False
        return success,theta
    y_5 = y_5/norm_y5
    norm_z5 = np.linalg.norm(vects[:,5])
    if norm_z5==0:
        success = False
        return success,theta
    z_5 = -vects[:,5]/norm_z5
    x_5 = np.cross(y_5,z_5)
    norm_x5 = np.linalg.norm(x_5)
    if norm_x5==0:
        success = False
        return success,theta
    x_5 = x_5/norm_x5
    R_w1 = R_e1.T@R_shoulder.T@np.concatenate((x_5.reshape((3,1)),y_5.reshape((3,1)),z_5.reshape((3,1))),axis=1)
    theta[12], theta[13], theta[14] = tr2rpy(R_w1)    
    
    #go from shoulder z axis to the opposite of the shoulder2->elbow2 vector
    #align the elbow y axis with the shoulder->shoulder vector crossed with elbow2->shoulder2
    y_6 = np.cross(shoulder2shoulder,-vects[:,6])
    norm_y6 = np.linalg.norm(y_6)
    if norm_y6==0:
        success = False
        return success,theta
    y_6 = y_6/norm_y6
    z_6 = -vects[:,6]
    norm_z6 = np.linalg.norm(z_6)
    if norm_z6==0:
        success = False
        return success,theta
    z_6 = z_6/norm_z6
    x_6 = np.cross(y_6,z_6)
    norm_x6 = np.linalg.norm(x_6)
    if norm_x6==0:
        success = False
        return success,theta
    x_6 = x_6/norm_x6
    R_e2 = R_shoulder.T@np.concatenate((x_6.reshape((3,1)),y_6.reshape((3,1)),z_6.reshape((3,1))),axis=1)
    theta[15], theta[16], theta[17] =  tr2rpy(R_e2)
    
    #align the elbow z axis with the elbow2->wrist2 vector
    #align the wrist y axis with the cross product of the elbow2->wrist2 vector and the shoulder2->elbow2 vector
    y_7 = np.cross(vects[:,7],vects[:,6])
    norm_y7 = np.linalg.norm(y_7)
    if norm_y7==0:
        success = False
        return success,theta
    y_7 = y_7/norm_y7
    norm_z7 = np.linalg.norm(vects[:,7])
    if norm_z7==0:
        success = False
        return success,theta
    z_7 = -vects[:,7]/norm_z7
    x_7 = np.cross(y_7,z_7)
    norm_x7 = np.linalg.norm(x_7)
    if norm_x7==0:
        success = False
        return success,theta
    x_7 = x_7/norm_x7
    R_w2 = R_e2.T@R_shoulder.T@np.concatenate((x_7.reshape((3,1)),y_7.reshape((3,1)),z_7.reshape((3,1))),axis=1)
    theta[18], theta[19], theta[20] =  tr2rpy(R_w2)   

    return success,theta

def wrap2pi(theta):
    over_pi = theta>np.pi
    under_pi = theta<-np.pi
    theta=theta-over_pi*2*np.pi+under_pi*2*np.pi
    return theta

def kalman_filter(theta,prev_estimate):
    #this function updates the kalman filter estimated state and predicted state with the
    #previous estimation and current joint angles
    #x_hat_n is the previous/current estimation
    #x_hat_np1 is the predicted state
    #state vector is 63 elements: 21 joint angles, 21 joint velocities, and 21 joint accelerations
    #angles are measured, velocities and accelerations are estimated
    error = theta-prev_estimate[:21] #estimation error
    state_estimate = prev_estimate + A*np.tile(error,(3,1)) #predicted state
    state_estimate[21:] = state_estimate[21:]-0.3*prev_estimate[21:] #velocity and accel. decay to zero when the error is zero
    #if the estimated state angles are outside of -pi to pi, 
    #then modify them to stay in range
    #state_estimate[:21]=wrap2pi(state_estimate[:21])
    next_state = F@state_estimate #estimated state based on predicted state
    #if the estimated state angles are outside of -pi to pi, 
    #then modify them to stay in range
    #next_state[:21]=wrap2pi(next_state[:21])
    
    return state_estimate, next_state

def forwardKinematics(theta,pelvis_coords,dimension):
    #function to compute the spine, shoulder, elbow, wrist, and head top locations
    #given the rpy at each joint.  The coordinates of the pelvis must be given since
    #other transformations start from that location.  
    #dimension[0] = spine length, [1]=neck/head length, [2] = shoulder to shoulder length
    #[3] = L upper arm length, [4] = L forearm length, [5]/[6] = R upper arm/forearm length
    points = np.zeros((3,15))
    points[:,0] = pelvis_coords
    t_spine = transl(pelvis_coords[0],pelvis_coords[1],pelvis_coords[2])@rpy2tr(theta[0:3])@transl(0,0,dimension[0])
    points[:,1] = transl(t_spine)
    t_head = t_spine@rpy2tr(theta[3:6])@transl(0,0,dimension[1])
    points[:,2] = transl(t_head)
    t_shoulders = t_spine@rpy2tr(theta[6:9])
    t_s1 = t_shoulders@transl(-0.5*dimension[2],0,0)
    points[:,3] = transl(t_s1)
    t_e1 = t_s1@rpy2tr(theta[9:12])@transl(0,0,-dimension[3])
    points[:,4] = transl(t_e1)
    t_w1 = t_e1@rpy2tr(theta[12:15])@transl(0,0,-dimension[4])
    points[:,5] = transl(t_w1)
    t_s2 = t_shoulders@transl(0.5*dimension[2],0,0)
    points[:,6] = transl(t_s2)
    t_e2 = t_s2@rpy2tr(theta[15:18])@transl(0,0,-dimension[5])
    points[:,7] = transl(t_e2)
    t_w2 = t_e2@rpy2tr(theta[18:21])@transl(0,0,-dimension[6])
    points[:,8] = transl(t_w2)
    return points

def realSenseSetup():
    # Configure depth and color streams of the intel realsense
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start the realsense pipeline
    pipeline = rs.pipeline()
    pipeline.start()

    # Create align object to align depth frames to color frames
    align = rs.align(rs.stream.color)

    # Get the intrinsics information for calculation of 3D point
    unaligned_frames = pipeline.wait_for_frames()
    frames = align.process(unaligned_frames)
    depth = frames.get_depth_frame()
    depth_intrinsic = depth.profile.as_video_stream_profile().intrinsics

    # Initialize the cubemos api with a valid license key in default_license_dir()
    skeletrack = skeletontracker(cloud_tracking_api_key="")
    joint_confidence = 0.2

     # Create window for initialisation
    window_name = "cubemos skeleton tracking with realsense D400 series"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)

    return pipeline, align, unaligned_frames, frames, depth, depth_intrinsic, window_name, skeletrack, joint_confidence

def findUnitVect(U1, U2):
    U12 = np.array([U2[0] - U1[0], U2[1] - U1[1], U2[2] - U1[2]])
    U12 = U12/np.linalg.norm(U12)

    if np.linalg.norm(U12) < 0.001:
        divByZero = True
    else:
        divByZero = False

    return U12, divByZero

def findVect(U1, U2):
    U12 = np.array([U2[0] - U1[0], U2[1] - U1[1], U2[2] - U1[2]])
    return U12

def transformToFixedFrame(points, angle):

    #Convert to homogeneous coordinates
    a,b = np.shape(points)
    points_H = np.zeros((b+1,a))
    for ii in range(3):
        points_H[ii,:] = points[:,ii]
    points_H[3,:] = np.ones((1,a))

    #Rotate about x to align with fixed
    R = np.array([[1,                0,                0, 0],
                [  0,  math.cos(angle), -math.sin(angle), 0],
                [  0,  math.sin(angle),  math.cos(angle), 0],
                [  0,                0,                0, 1]])
    #Translate to fixed
    T = np.array([[1, 0, 0,  -0.3],
                [  0, 1, 0, -2.5],
                [  0, 0, 1, 0.8],
                [  0, 0, 0, 1]])
    #Transformationof Points
    points_H = R @ T @ points_H

    #Convert back to standard coordinates
    for ii in range(3):
        points[:,ii] = points_H[ii]

    return points

def transformMat(Ux, Uy, Uz, P):
    #Find rotation between fixed basis and torso basis then translate to point P
    T_ba = np.zeros((4,4))
    T_ba[0:3,0] = Ux
    T_ba[0:3,1] = Uy
    T_ba[0:3,2] = Uz
    T_ba[3,3] = 1
    T_ba[0:3,3] = P

    return T_ba

def torsoTransform(mid_point, norm, boundary_points, boundary_points_C, P):

    #Establish coordinate system from torso and waist, centered at torso
    Ux, divByZero = findUnitVect(P, mid_point)
    Uz = norm
    Uy = np.cross(Uz, Ux)

    #Define shoulder, neck, and pelvis in new coordinate system
    #--------------------------------------------------------------------------------------
    #Make proportional to rest of coordinates
    neck_left    = np.array([[-0.07], [0.08],  [0],      [1]])
    neck_right   = np.array([[-0.07], [-0.08], [0],      [1]])
    neck_front   = np.array([[ 0],    [0],     [0.1],    [1]])
    neck_back    = np.array([[ 0],    [0],     [-0.1],   [1]])
    pelvis       = np.array([[ 0.6],  [0],     [0],      [1]])
    pelvis_front = np.array([[ 0.6],  [0],     [0.08],   [1]])
    pelvis_back  = np.array([[ 0.6],  [0],     [-0.08],  [1]])
    #--------------------------------------------------------------------------------------

    #Transport these points to home coordinate sytem and add to boundary_points
    T_ba = transformMat(Ux, Uy, Uz, P)

    for ii in range(3):
        boundary_points[ii,10]  = (T_ba @  neck_left)[ii]
        boundary_points[ii,7]   = (T_ba @  neck_right)[ii]
        boundary_points_C[ii,1] = (T_ba @  neck_front)[ii]
        boundary_points_C[ii,4] = (T_ba @  neck_back)[ii]
        boundary_points_C[ii,0] = (T_ba @  pelvis_front)[ii]
        boundary_points_C[ii,5] = (T_ba @  pelvis_back)[ii]
        boundary_points[ii,22]  = (T_ba @  pelvis)[ii]

    return boundary_points, boundary_points_C, divByZero
    
def jointPoint(norm, joint, A, B):

    #Get unit vectors pointing away from joint to the next keypoint
    Ua, divByZero = findUnitVect(A, joint)
    Ub, divByZero = findUnitVect(B, joint)

    #Direction vector that points to inside for Left side and Outside for right side
    Udir = np.cross(norm, Ua)

    #Project vectors into the torso plane by subtracting out portion orthoganol to plane
    Ua_2D = Ua - np.dot(Ua,norm) * norm
    Ub_2D = Ub - np.dot(Ub,norm) * norm

    #Find bisecting vector
    b = Ua_2D/np.linalg.norm(Ua_2D) + Ub_2D/np.linalg.norm(Ub_2D)
    b = b/np.linalg.norm(b)

    #Find boundary points equally spaced on either side of the joint point along bisection line
    P1 = 0.05 * b + joint
    P2 = - 0.05 * b + joint
    O1 = 0.05 * norm + joint
    O2 = - 0.05 * norm + joint

    return P1, P2, O1, O2, Udir, divByZero

def assignJoint(inside,outside,P1,P2,Udir,boundary_points,ortho,back,front,O1,O2, Odir):
    
    if np.dot(P1,Udir) > np.dot(P2,Udir):
        boundary_points[:,inside] = P1
        boundary_points[:,outside] = P2
    else:
        boundary_points[:,inside] = P2
        boundary_points[:,outside] = P1

    if np.dot(O1,Odir) > np.dot(O2,Odir):
        ortho[:,front] = O1
        ortho[:,back] = O2
    else:
        ortho[:,front] = O2
        ortho[:,back] = O1
    
    return boundary_points, ortho

def handFootPoints(norm, joint, A):
    #Get direction along skeleton link
    Ua, divByZero = findUnitVect(joint,A)

    #Get direction towards a boundary point
    Ub = np.cross(norm,Ua)/np.linalg.norm(np.cross(norm,Ua))

    #Add and subtract scaled Ub to obtain points
    P1 = 0.05 * Ub + joint
    P2 = - 0.05 * Ub + joint
    O1 = 0.05 * norm + joint
    O2 = - 0.05 * norm + joint 

    return P1, P2, O1, O2, Ub, divByZero

def findMissingPoints(joint_location):
    #Run through joint_location and find missing points. Return their indices in an array 
    missingList = []
    for ii in range(len(joint_location)):
        if joint_location[ii,0] == 0.0 and joint_location[ii,1] == 0.0 and joint_location[ii,2] == 0.0:
            missingList.append(ii)

    missingArray = np.array(missingList)
    return missingArray

def headMidPoint(joint_location, missingArray):
    inopperable = False
    if not (0 in missingArray):
        return joint_location[0,:], inopperable
    elif not (16 in missingArray or 17 in missingArray):
        return np.array([(joint_location[16,0] + joint_location[17,0])/2, (joint_location[16,1] + joint_location[17,1])/2, (joint_location[16,2] + joint_location[17,2])/2]), inopperable
    elif not (14 in missingArray or 15 in missingArray):
        #Create vector from center of head to nose
        return np.array([(joint_location[14,0] + joint_location[15,0])/2, (joint_location[14,1] + joint_location[15,1])/2, (joint_location[14,2] + joint_location[15,2])/2]), inopperable
    elif not (16 in missingArray):
        return joint_location[16,0], inopperable
    elif not (17 in missingArray):
        return joint_location[17,0], inopperable
    elif not (14 in missingArray):
        return joint_location[14,0], inopperable
    elif not (15 in missingArray):
        return joint_location[15,0], inopperable
    else:
        inopperable = True
        return np.array([0,0,0]), inopperable

def headPoints(boundary_points, boundary_points_C, joint_location, missingArray):

    divByZero = False
    error_head = 0.7

    #Assuming a point has already been calculated for the nose
    #CASE 1: Left Ear Detected
    if not (17 in missingArray):
        Uhoriz, divByZero1 = findUnitVect(joint_location[0,:],joint_location[17,:])
    #CASE 2: Right Ear Detected
    elif not (16 in missingArray):
        Uhoriz, divByZero1 = findUnitVect(joint_location[16,:],joint_location[0,:])
    #CASE 3: Both eyes detected
    elif not (14 in missingArray or 15 in missingArray):
        Uhoriz, divByZero1 = findUnitVect(joint_location[14,:],joint_location[15,:])
    #CASE 4: Use both calculated points on neck
    else:
        Uhoriz, divByZero1 = findUnitVect(boundary_points[:,7],boundary_points[:,10])

    #Establish Head Coordinate System
    Hvect, divByZero2 = findUnitVect(joint_location[0,:], joint_location[1,:])
    norm = np.cross(Hvect, Uhoriz)
    Uvert = np.cross(norm,Uhoriz)

    #Establish Boundary Point Locations
    boundary_points[:, 8] = joint_location[0,:] + Uvert * 0.15 - Uhoriz * 0.1
    boundary_points[:, 9] = joint_location[0,:] + Uvert * 0.15 + Uhoriz * 0.1
    width = np.linalg.norm(findVect( boundary_points[:, 8],  boundary_points[:, 9] ))/2
    boundary_points_C[:, 2] = joint_location[0,:] + Uvert * 0.15 + norm * width
    boundary_points_C[:, 3] = joint_location[0,:] + Uvert * 0.15 - norm * width

    #Perform a few checks to filter out singularities
    #Check for any caught singularities
    if divByZero1 or divByZero2:
        divByZero = True

    #Check calculated points
    if abs(np.linalg.norm(findVect(joint_location[1,:],boundary_points[:,8]))) > error_head or abs(np.linalg.norm(findVect(joint_location[1,:],boundary_points[:,9]))) > error_head:
        divByZero = True

    return boundary_points, boundary_points_C, divByZero

def filterOutliers(joint_location, missingArray):
    inopperable = False

    #Abort this frame if both shoulders are gone
    if (2 in missingArray) and (5 in missingArray):
        inopperable = True
    
    #Eliminate noisy frames from consideration by making sure problem points are too far from each other
    #Problem points include: Feet, Shoulders, hands
    error_shoulder = 0.55
    error_hand = 0.7
    error_foot = 0.8
    error_elbow = 0.7
    error_knee = 0.8
    error_head = 0.7

    #check left and right shoulders
    if abs(np.linalg.norm(findVect(joint_location[1,:],joint_location[2,:]))) > error_shoulder or abs(np.linalg.norm(findVect(joint_location[1,:],joint_location[5,:]))) > error_shoulder:
        inopperable = True

    #check left and right shoulders
    if abs(np.linalg.norm(findVect(joint_location[1,:],joint_location[0,:]))) > error_head:
        inopperable = True

    #Check hand distance from elbows
    if abs(np.linalg.norm(findVect(joint_location[3,:],joint_location[4,:]))) > error_hand or abs(np.linalg.norm(findVect(joint_location[6,:],joint_location[7,:]))) > error_hand:
        #Only check if the joints have been detected
        if not(3 in missingArray) and not(4 in missingArray) and not(5 in missingArray) and not(6 in missingArray):
            inopperable = True
            
    #Check feet distance to knee
    if abs(np.linalg.norm(findVect(joint_location[9,:],joint_location[10,:]))) > error_foot or abs(np.linalg.norm(findVect(joint_location[12,:],joint_location[13,:]))) > error_foot:
        #Only check if the joints have been detected
        if not(9 in missingArray) and not(10 in missingArray) and not(12 in missingArray) and not(13 in missingArray):
            inopperable = True
            
    #Check elbow distance to shoulder
    if abs(np.linalg.norm(findVect(joint_location[3,:],joint_location[2,:]))) > error_elbow or abs(np.linalg.norm(findVect(joint_location[6,:],joint_location[5,:]))) > error_elbow:
        #Only check if the joints have been detected
        if not(2 in missingArray) and not(3 in missingArray) and not(5 in missingArray) and not(6 in missingArray):
            inopperable = True
            
    #Check knee distance to hip
    if abs(np.linalg.norm(findVect(joint_location[8,:],joint_location[9,:]))) > error_knee or abs(np.linalg.norm(findVect(joint_location[11,:],joint_location[12,:]))) > error_knee:
        #Only check if the joints have been detected
        if not(8 in missingArray) and not(9 in missingArray) and not(11 in missingArray) and not(12 in missingArray):
            inopperable = True

    #If the torso or hips are not detected, this frame is inopperable
    requiredPoints = np.array([1,8,11])
    for ii in requiredPoints:
        if ii in missingArray:
            inopperable = True

    return inopperable

def getEndParent(node, missingArray):

    #If node and parent are missing default to 1
    if (node in missingArray) and ((node -1) in missingArray):
        return 1

    #If the node is missing, the parent node should be skipped
    if node in missingArray:
        if not((node - 2) in missingArray):
            return node - 2
        else:
            return 1

    #Node exists but other data may be missing
    if not((node - 1) in missingArray):
        return node - 1
    elif not((node - 2) in missingArray):
        return node - 2
    else:
        return 1

def getJointParent(node, missingArray):

    #If the node is missing, the parent node should be skipped
    if node in missingArray:
        return 1

    #Node exist but other data may be missing
    if not((node - 1) in missingArray):
        return node - 1
    else:
        return 1

def getChildJoint(node, missingArray):
    if not((node + 1) in missingArray):
        return node + 1
    else:
        return getJointParent(node, missingArray)

def getChildShoulder(node, missingArray):
    if not((node + 1) in missingArray):
        return node + 1
    elif not((node + 2) in missingArray):
        return node + 2
    else:
        return getJointParent(node, missingArray)

def setToParents(skel, missingArray):
    #Check Elbows and Knees. If missing, set equal to shoulders and knees
    #Left Elbow
    if 6 in missingArray:
        skel[6] = skel[5]

    #Right Elbow
    if 3 in missingArray:
        skel[3] = skel[2]

    #Left Knee
    if 12 in missingArray:
        skel[12] = skel[11]

    #Right Knee
    if 9 in missingArray:
        skel[9] = skel[8]

    #Check hands and feet. If missing, set equal to elbows and knees
    #Left Hand
    if 7 in missingArray:
        skel[7] = skel[6]

    #Right Hand
    if 4 in missingArray:
        skel[4] = skel[3]

    #Left Foot
    if 13 in missingArray:
        skel[13] = skel[12]

    #Right Foot
    if 10 in missingArray:
        skel[10] = skel[9]

    return skel

def boundaryPoints(joint_location, missingArray):
    
    #Initialize boundary points to max boundary. Also initialize ortho curves for arms, legs, and core
    boundary_points_RA = np.zeros((3,6))
    boundary_points_LA = np.zeros((3,6))
    boundary_points_RL = np.zeros((3,6))
    boundary_points_LL = np.zeros((3,6))
    boundary_points_C  = np.zeros((3,6))
    boundary_points    = np.zeros((3,27))

    #Deterine the general orientation of the skeleton by defining a plane between the two hip points and the torso point
    #Find two vectors with the three points to define the plane
    Uah, divByZero = findUnitVect(joint_location[8,:], joint_location[1,:])
    Uak, divByZero = findUnitVect(joint_location[11,:], joint_location[1,:])
    
    #Use cross product to get norm
    norm = np.cross(Uah, Uak)/np.linalg.norm(np.cross(Uah, Uak))

    #If bare minimum points are present, procede
    success = True
                
    #Find mid point of waist
    mid_point = np.array([(joint_location[8,0] + joint_location[11,0])/2, (joint_location[8,1] + joint_location[11,1])/2, (joint_location[8,2] + joint_location[11,2])/2])

    #Waist points are kept as is
    boundary_points[:, 0] = joint_location[8,:]
    boundary_points[:, 17] = joint_location[11,:]


    #Boundary points of torso 
    boundary_points, boundary_points_C, divByZero = torsoTransform(mid_point, norm, boundary_points, boundary_points_C, joint_location[1,:])
    
    #Boundary points of Head (boundary points contains filter for head points for conveinence)
    boundary_points, boundary_points_C, divByZero = headPoints(boundary_points, boundary_points_C, joint_location, missingArray)

    #Find mid point of waist and top of each leg, then ortho points for legs
    right_mid = np.array([(joint_location[8,0] + boundary_points[0,22])/2, (joint_location[8,1] + boundary_points[1,22])/2, (joint_location[8,2] + boundary_points[2,22])/2])
    left_mid = np.array([(joint_location[11,0] + boundary_points[0,22])/2, (joint_location[11,1] + boundary_points[1,22])/2, (joint_location[11,2] + boundary_points[2,22])/2])
    boundary_points_RL[:,0] = right_mid - norm * 0.05
    boundary_points_RL[:,5] = right_mid + norm * 0.05
    boundary_points_LL[:,0] = left_mid - norm * 0.05
    boundary_points_LL[:,5] = left_mid + norm * 0.05


    #Boundary points of left foot
    parent = getEndParent(13, missingArray)
    P1, P2, O1, O2, Udir, divByZero = handFootPoints(norm, joint_location[13,:], joint_location[parent,:])
    boundary_points, boundary_points_LL =  assignJoint(20,19,P1,P2,Udir,boundary_points,boundary_points_LL,2,3,O1,O2,norm)


    #Boundary points of right foot
    parent = getEndParent(10, missingArray)
    P1, P2, O1, O2, Udir, divByZero = handFootPoints(norm, joint_location[10,:], joint_location[parent,:])
    boundary_points, boundary_points_RL =  assignJoint(25,24,P1,P2,Udir,boundary_points,boundary_points_RL,2,3,O1,O2,norm)


    #Boundary points of left hand
    parent = getEndParent(7, missingArray)
    P1, P2, O1, O2, Udir, divByZero = handFootPoints(norm, joint_location[7,:], joint_location[parent,:])
    boundary_points, boundary_points_LA =  assignJoint(14,13,P1,P2,Udir,boundary_points,boundary_points_LA,2,3,O1,O2,norm)


    #Boundary points of right hand
    parent = getEndParent(4, missingArray)
    P1, P2, O1, O2, Udir, divByZero = handFootPoints(norm, joint_location[4,:], joint_location[parent,:])
    boundary_points, boundary_points_RA =  assignJoint(4,3,P1,P2,Udir,boundary_points,boundary_points_RA,2,3,O1,O2,norm)


    #Boundary points of left knee. Use distance from pelvis to decide which point goes to outside of joint
    parent = getJointParent(12, missingArray)
    child = getChildJoint(12, missingArray)
    P1, P2, O1, O2, Udir, divByZero = jointPoint(norm, joint_location[12,:], joint_location[parent,:], joint_location[child,:])
    boundary_points, boundary_points_LL =  assignJoint(18,21,P1,P2,Udir,boundary_points,boundary_points_LL,1,4,O1,O2,norm)


    #Boundary points of right knee. Use distance from pelvis to decide which point goes to outside of joint
    parent = getJointParent(9, missingArray)
    child = getChildJoint(9, missingArray)
    P1, P2, O1, O2, Udir, divByZero = jointPoint(norm, joint_location[9,:], joint_location[parent,:], joint_location[child,:])
    boundary_points ,boundary_points_RL =  assignJoint(23,26,P1,P2,Udir,boundary_points,boundary_points_RL,1,4,O1,O2,norm)


    #Boundary points of left elbow
    parent = getJointParent(6, missingArray)
    child = getChildJoint(6, missingArray)
    P1, P2, O1, O2, Udir, divByZero = jointPoint(norm, joint_location[6,:], joint_location[parent,:], joint_location[child,:])
    boundary_points, boundary_points_LA =  assignJoint(12,15,P1,P2,Udir,boundary_points,boundary_points_LA,1,4,O1,O2,norm)


    #Boundary points of right elbow
    parent = getJointParent(3, missingArray)
    child = getChildJoint(3, missingArray)
    P1, P2, O1, O2, Udir, divByZero = jointPoint(norm, joint_location[3,:], joint_location[parent,:], joint_location[child,:])
    boundary_points, boundary_points_RA =  assignJoint(2,5,P1,P2,Udir,boundary_points,boundary_points_RA,1,4,O1,O2,norm)


    #Boundary points of left shoulder. Use distance from pelvis to decide which point goes to outside of joint
    child = getChildShoulder(5, missingArray)
    P1, P2, O1, O2, Udir, divByZero = jointPoint(norm, joint_location[5,:], joint_location[1,:], joint_location[child,:])
    boundary_points, boundary_points_LA =  assignJoint(11,16,P1,P2,Udir,boundary_points,boundary_points_LA,0,5,O1,O2,norm)


    #Boundary points of right shoulder. Use distance from pelvis to decide which point goes to outside of joint
    child = getChildShoulder(2, missingArray)
    P1, P2, O1, O2, Udir, divByZero = jointPoint(norm, joint_location[2,:], joint_location[1,:], joint_location[child,:])
    boundary_points, boundary_points_RA =  assignJoint(1,6,P1,P2,Udir,boundary_points,boundary_points_RA,0,5,O1,O2,norm)

    if (divByZero):
        success = False

    #Perform final check to make sure no points have been set to [0,0,0]
    zero = np.zeros([3,1])
    for ii in range(27):
        if (boundary_points[:,ii] == zero).all():
            success = False

    for jj in range(6):
        if (boundary_points_C[:,jj] == zero).all() or (boundary_points_RA[:,jj] == zero).all() or (boundary_points_LA[:,jj] == zero).all() or (boundary_points_RL[:,jj] == zero).all() or (boundary_points_LL[:,jj] == zero).all():
            success = False 
    
    return boundary_points, boundary_points_C, boundary_points_RA, boundary_points_LA, boundary_points_RL, boundary_points_LL, success

def visualLive(patches, collisionPoints):
    #Function to plot swept surfaces during use
    plt.ion()
    plt.show()
    colmap = plt.get_cmap('jet')
    minn, maxx = patches[3,:].min(), patches[3,:].max()
    norm = cm.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap=colmap)
    fig = plt.figure()
    ax = Axes3D(fig)
    #Plot
    if collisionPoints.size > 0:
        ax.scatter(collisionPoints[:,0], collisionPoints[:,1], collisionPoints[:,2], c='red', s=100)
        #Plot sweeps with transparency
        ax.scatter(patches[0,:], patches[1,:], patches[2,:], c=m.to_rgba(patches[3,:]), alpha=0.1)
    else:
        #Plot sweeps with no transparency
        ax.scatter(patches[0,:], patches[1,:], patches[2,:], c=m.to_rgba(patches[3,:]))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.axes.set_xlim3d(left=-1, right=1) 
    ax.axes.set_zlim3d(bottom=0, top=2) 
    ax.axes.set_ylim3d(bottom=-1, top=1) 
    m.set_array(patches[3,:])
    plt.draw()
    plt.pause(0.001)

def visual(point_cloud, collisionPoints):
    #Function to plot swept surfaces after use
    colmap = plt.get_cmap('jet')
    minn, maxx = point_cloud[3,:].min(), point_cloud[3,:].max()
    norm = cm.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap=colmap)
    fig = plt.figure()
    ax = Axes3D(fig)
    #Plot
    if collisionPoints.size > 0:
        ax.scatter(collisionPoints[:,0], collisionPoints[:,1], collisionPoints[:,2], c='red', s=100, marker='.')
        #Plot sweeps with transparency
        ax.scatter(point_cloud[0,:], point_cloud[1,:], point_cloud[2,:], c=m.to_rgba(point_cloud[3,:]), alpha=0.1, marker='.')
    else:
        #Plot sweeps with no transparency
        ax.scatter(point_cloud[0,:], point_cloud[1,:], point_cloud[2,:], c=m.to_rgba(point_cloud[3,:]), marker='.')
    
    m.set_array(point_cloud[3,:])
    fig.colorbar(m,label='Time')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.axes.set_xlim3d(left=-1.5, right=1.5) 
    ax.axes.set_zlim3d(bottom=0, top=3) 
    ax.axes.set_ylim3d(bottom=-1.5, top=1.5) 
    plt.show()

def sweepSurfaceRobot(jointTraj_rob, Traj_rob, grid_spacing):

    #Determine XYZ position of each joint at each pose for robot
    a = 12
    b = len(jointTraj_rob[0])
    points = np.zeros(shape=(b,4,a))
    points_shift = np.zeros(shape=(b,4,a))

    for ii in range(0,len(jointTraj_rob[0])):
        points[ii,:,:] = JointPoints(jointTraj_rob[:,ii], a)
        points_shift[ii,:,:] = JointPoints_shift(jointTraj_rob[:,ii], a)

    #Determine the area swept out by robot. Data stored in XYZ coordinates within patches
    patches = generateSurface(points,Traj_rob,grid_spacing)
    patches_shift = generateSurface(points_shift,Traj_rob,grid_spacing)
    
    #Set Boundary conditions for robot
    IC = 0
    BC_f_r = boundary_conditions_robot(points[-1,:,:], points_shift[-1,:,:],Traj_rob,grid_spacing,IC)

    #Add all components of robot sweep together
    robot = np.zeros((6,len(patches[0])+len(patches_shift[0])+len(BC_f_r[0])))
    for ii in range(6):
        robot[ii,:] = np.concatenate((patches[ii],patches_shift[ii],BC_f_r[ii]), axis=None)

    return robot

def sweepSurfaceHuman(human, human_C, human_LA, human_LL, human_RA, human_RL, Trajectory, grid_spacing):
    #This method invokes all the necessary functions after loading the joint trajectories
    #Determine the area swept out by human. Data stored in XYZ coordinates within patches
    patches_hum = generateSurface(human,Trajectory,grid_spacing)
    patches_shift_C  = generateSurface(human_C,Trajectory,grid_spacing)
    patches_shift_LA = generateSurface(human_LA,Trajectory,grid_spacing)
    patches_shift_RA = generateSurface(human_RA,Trajectory,grid_spacing)
    patches_shift_LL = generateSurface(human_LL,Trajectory,grid_spacing)
    patches_shift_RL = generateSurface(human_RL,Trajectory,grid_spacing)

    
    #Set Boundary conditions for human
    IC = 0
    BC_f_h = boundary_conditions_human(human[-1,:,:], human_C[-1,:,:],human_LA[-1,:,:],human_RA[-1,:,:],human_LL[-1,:,:],human_RL[-1,:,:],Trajectory,grid_spacing,IC)
    
    #Add all components of human sweep together
    human = np.zeros((6,len(patches_hum[0])+len(patches_shift_C[0])+len(patches_shift_LA[0])+len(patches_shift_RA[0])+len(patches_shift_LL[0])+len(patches_shift_RL[0])+len(BC_f_h[0])))
    for ii in range(6):
        human[ii,:] = np.concatenate((patches_hum[ii],patches_shift_C[ii],patches_shift_LA[ii],patches_shift_RA[ii],patches_shift_LL[ii],patches_shift_RL[ii],BC_f_h[ii]), axis=None)
    
    return human

def swapYZ(boundary_points):
    Y = boundary_points[:,2].copy()
    Z = -boundary_points[:,1].copy()
    boundary_points[:,2] = Z
    boundary_points[:,1] = Y
    return boundary_points

def getCamData():
    #------------------------------------------------------------------------------------------------------------------
    #Currently only works for one human, will get confused with two, so I'm limiting to one till we can upgrade
    #------------------------------------------------------------------------------------------------------------------
    frames_rejected = 0
    boundTime = 0.0
    kalTime = 0.0
    skelTime = 0.0
    global index    

    while True:
        
        start_time = time.time()
        if testing:
            time.sleep(0.06)
            #TESTING DATA
            joint_location = humanmotion[index]
            index +=1
            skeletons = joint_location 
        else:
            #Listen to pipeline for new frame 
            unaligned_frames = pipeline.wait_for_frames()
            frames = align.process(unaligned_frames)
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color.get_data())
            
            # perform inference and update the tracking id
            skeletons = skeletrack.track_skeletons(color_image)
            # Make sure at least 1 skeleton is in frame
            if len(skeletons) < 1:
                continue
            # render the skeletons on top of the acquired image and display it
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            cmr.render_result(skeletons, color_image, joint_confidence)
            joint_location = render_ids_3d(color_image, skeletons, depth, depth_intrinsic, joint_confidence)
        

        
        skelTime = skelTime + (time.time()-start_time)
        if len(skeletons) > 0:
            start_time = time.time()
            #Limiting to one
            joint_location = joint_location[0,:,:]

            #Find missing data points
            missingArray = findMissingPoints(joint_location)
    
            #Convert to workspace coordinate system
            joint_location = swapYZ(joint_location)
            joint_location = transformToFixedFrame(joint_location, -10*math.pi/180)

            #Create nose to allow Kalman filter to use data from back view of human
            joint_location[0,:], inopperable = headMidPoint(joint_location, missingArray)

            #Alter data Indices then pass to Kalman Filter, then restore original indices
            kalPoints = mapSkelToKalman(joint_location)
            kalPoints = kalmanFilter(kalPoints)
            joint_location = mapKalmanToSkel(kalPoints, joint_location)
            kalTime =np.max([kalTime, (time.time()-start_time)])
            
            #Set any necessary parent joints
            joint_location = setToParents(joint_location, missingArray)

            #Perform a series of checks to asses if the frame's data is of good quality
            inopperable = filterOutliers(joint_location, missingArray)

            missingArray = np.delete(missingArray,[0])
            if inopperable:
                frames_rejected+=1
                continue

            #Find boundary points for one skeleton
            start_time = time.time()
            boundary_points, boundary_points_C, boundary_points_RA, boundary_points_LA, boundary_points_RL, boundary_points_LL, success = boundaryPoints(joint_location, missingArray)
            boundTime = np.max([boundTime, (time.time()-start_time)])

            if success:
                return  boundary_points, boundary_points_C, boundary_points_RA, boundary_points_LA, boundary_points_RL, boundary_points_LL, frames_rejected, skelTime, kalTime, boundTime
            else:
                frames_rejected+=1
            if not testing:
                cv2.imshow(window_name, color_image)
                if cv2.waitKey(1) == 27:
                    break
         
if __name__ == "__main__":

    #Opperation mode (True for pre saved data)
    testing = True

    #Global Parameters
    grid_spacing = 0.05
    dt = 1/20 #time step
    alpha = 0.5 #gain for change in angle
    beta = 0.5 #gain for change in velocity
    gamma = 0.05 #gain for change in acceleration
    dims = np.array([])
    actuals = np.array([])
    use_intel =True #boolean to using Intel or MPII database skeleton
    ang_success=True #boolean to indicate success of computing joint angles
    init_kalman = True

     #F, G, and A are matrices for use in the Kalman Filter
    F=np.concatenate((np.concatenate((np.eye(21),dt*np.eye(21),0.5*dt**2*np.eye(21)),axis=1),np.concatenate((np.zeros((21,21)),np.eye(21),dt*np.eye(21)),axis=1),np.concatenate((np.zeros((21,21)),np.zeros((21,21)),np.eye(21)),axis=1)))
    G=np.array([[(1-alpha)*np.eye(21),np.zeros((21,21)),np.zeros((21,21))],[np.zeros((21,21)),(1-beta/dt)*np.eye(21),np.zeros((21,21))],[np.zeros((21,21)),np.zeros((21,21)),(1-2*gamma/dt**2)*np.eye(21)]])
    A = np.concatenate((alpha*np.ones((21,1)), beta/dt*np.ones((21,1)), 2*gamma/dt**2*np.ones((21,1))))
    #construct a low pass 2nd order buttorworth filter to smooth the computed joint angles
    b, a = butter(2, 0.20, btype='low', analog=False)
    x_hat = F@np.zeros((63,1)) #initialize the state estimates
    x_hat_next = x_hat #init state predictions

    #------------------------------------------------------------------------------------------------------------
    #TESTING DATA
    #------------------------------------------------------------------------------------------------------------
    if testing:
        index = 0
        with open('HumanMotionBad.pickle', 'rb') as handle:
            humanmotion = pickle.load(handle)

    #OFFLINE PORTION
    #------------------------------------------------------------------------------------------------------------
    #Perform compilation of all @jit decorated functions by passing in a short sample trajectory
    Comp_human = np.array([1.0*np.zeros((3,27)) , 1.0*np.ones((3,27))])
    Comp_human_shift = np.array([1.0*np.zeros((3,6)) , 1.0*np.ones((3,6))])
    Comp_joint_traj = np.array([[0.0,0.5],[0.0,0.5],[0.0,0.5],[0.0,0.5],[0.0,0.5],[0.0,0.5],[0.0,0.5]])
    Comp_traj = np.array([[0.0,0.5,],[1.0,2.0]])
    pc_1 = sweepSurfaceRobot(Comp_joint_traj, Comp_traj, 0.1) 
    pc_2 = sweepSurfaceHuman(Comp_human, Comp_human_shift, Comp_human_shift, Comp_human_shift, Comp_human_shift, Comp_human_shift, Comp_traj, 0.1) 
    collisionCheck(pc_1,pc_2,0.1)
    
    #Load initial robot plans: These are in format that ROS will output
    with open('robotSchedule_iso.pickle', 'rb') as handle:
        robotSchedule = pickle.load(handle)
    robotScheduleSweep = defaultdict(list)
    robotScheduleTime = defaultdict(list)

    #Conduct the Offline Trajectory Sweeps for Robot
    for ii in range(int(len(robotSchedule))):
        jointTrajectory = robotSchedule[ii+1][0]
        Trajectory = robotSchedule[ii+1][1]
        robotScheduleSweep[ii] = sweepSurfaceRobot(jointTrajectory, Trajectory, 0.01)
        robotScheduleTime[ii] = robotSchedule[ii+1][1][0][-1]

    #Set up Real Sense Camera  
    if not testing:
        pipeline, align, unaligned_frames, frames, depth, depth_intrinsic, window_name, skeletrack, joint_confidence = realSenseSetup()

    #------------------------------------------------------------------------------------------------------------
    #ONLINE PORTION
    #------------------------------------------------------------------------------------------------------------
    #Dictionary to hold environment sweeps
    env_sweeps = defaultdict(list)

    #Initialize storage for human boundary curves (initialize to hold two frames of data)
    human = np.zeros((2, 3, 27))
    human_C = np.zeros((2, 3, 6))
    human_RA = np.zeros((2, 3, 6))
    human_LA = np.zeros((2, 3, 6))
    human_RL = np.zeros((2, 3, 6))
    human_LL = np.zeros((2, 3, 6))
    Traj_hum = np.zeros((2,2))

    #Start Timer
    SkelTime = np.array([])
    FramesRejected = np.array([])
    SweepTime = np.array([])
    ColTime = np.array([])
    KalTime = np.array([])
    BoundTime = np.array([])
    TotalTime = np.array([])
    start_time = 0
    human_pc = np.array([])
    combined_hum_pc = np.array([])

    #Step through each planned robot sweep
    for ii in range(int(len(robotScheduleSweep))):

        #Load current robot sweep
        robot_pc = robotScheduleSweep[ii]
        end_time = robotScheduleTime[ii]

        #Set an initial position for the human
        #Form boundary curves and insert into storage array
        human[0,:,:], human_C[0,:,:], human_RA[0,:,:], human_LA[0,:,:], human_RL[0,:,:], human_LL[0,:,:], frames_rejected, skelTime, kalTime, boundTime = getCamData()
        Traj_hum[0,0] = end_time*ii/int(len(robotScheduleSweep))
        Traj_hum[1,0] = 1
        Traj_hum[1,1] = 2

        #During execution of sweep, check human for imminent collisions
        startTask = end_time*ii/int(len(robotScheduleSweep))
        start = True
        while True:
            ttime = time.time()
            #Aquisition Skeleton Tracking SDK data
            #Form boundary curves and insert into storage array
            human[1,:,:], human_C[1,:,:], human_RA[1,:,:], human_LA[1,:,:], human_RL[1,:,:], human_LL[1,:,:], frames_rejected, skelTime, kalTime, boundTime = getCamData()
            FramesRejected = np.concatenate((FramesRejected,frames_rejected), axis=None)

            #Construct a sweep from the boundary curves
            sweep_time = time.time()
            Traj_hum[0,1] = startTask
            startTask += 0.06
            human_pc = sweepSurfaceHuman(human, human_C, human_LA, human_LL, human_RA, human_RL, Traj_hum, grid_spacing)

            #Set new boundary curves to old ones
            human[0,:,:] = np.copy(human[1,:,:])
            human_C[0,:,:] = np.copy(human_C[1,:,:])
            human_RA[0,:,:] = np.copy(human_RA[1,:,:]) 
            human_LA[0,:,:] = np.copy(human_LA[1,:,:]) 
            human_RL[0,:,:] = np.copy(human_RL[1,:,:]) 
            human_LL[0,:,:] = np.copy(human_LL[1,:,:])
            Traj_hum[0,0] = np.copy(Traj_hum[0,1])
            SweepTime = np.concatenate((SweepTime,time.time()-sweep_time), axis=None)
            #------------------------------------------------------------------------------------
            #Store human point clouds
            if not start:
                new_combined_hum_pc = np.zeros((4,len(combined_hum_pc[0])+len(human_pc[0])))
                for kk in range(4):
                    new_combined_hum_pc[kk,:] = np.concatenate((combined_hum_pc[kk],human_pc[kk]), axis=None)
                combined_hum_pc = np.copy(new_combined_hum_pc)
            else:
                combined_hum_pc = np.copy(human_pc)
                start = False
            #------------------------------------------------------------------------------------
            #Perform a collision check
            col_time = time.time()
            collision, segments, collisionPoints = collisionCheck(robot_pc,human_pc,grid_spacing)
            if collision:
                print("Collision at segment(s) ", segments)
            else:
                print('...')
            ColTime = np.concatenate((ColTime,time.time()-col_time), axis=None)

            TotalTime = np.concatenate((TotalTime,time.time()-ttime), axis=None)
            #------------------------------------------------------------------------------------
            #Plot cycle's prediction
            
            combined_pc = np.zeros((4,len(robot_pc[0])+len(human_pc[0])))
            for jj in range(4):
                combined_pc[jj,:] = np.concatenate((robot_pc[jj],human_pc[jj]), axis=None)
            visual(combined_pc, collisionPoints)
            
            #------------------------------------------------------------------------------------

        #------------------------------------------------------------------------------------
        #Combine robot and human for plot
        combined_pc = np.zeros((4,len(robot_pc[0])+len(combined_hum_pc[0])))
        for jj in range(4):
            combined_pc[jj,:] = np.concatenate((robot_pc[jj],combined_hum_pc[jj]), axis=None)
        env_sweeps[ii].append(combined_pc)
        env_sweeps[ii].append(collisionPoints)
        #------------------------------------------------------------------------------------
    if not testing:
        pipeline.stop()
        cv2.destroyAllWindows()
    
    #Plot all sweeps
    for ii in range(len(env_sweeps)):
        visual(env_sweeps[ii][0], env_sweeps[ii][1])
    
   