
#!/usr/bin/env my_pymc_env
from numba.core.typing.templates import bound_function
from skeletontracker import skeletontracker
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from testing_data import *
from kinematics import *
from generate_surface import *
from collision_check import collisionCheck
from collections import defaultdict
import numpy as np
from collections import defaultdict
from spatialmath.base import * #Peter Corke Spatial Math Toolbox

def swapYZ(boundary_points):
    Y = boundary_points[:,2].copy()
    Z = -boundary_points[:,1].copy()
    boundary_points[:,2] = Z
    boundary_points[:,1] = Y
    return boundary_points

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

def kalmanFilter(points):
    global x_hat
    global x_hat_next
    dims = np.array([np.linalg.norm(points[:,1]-points[:,0]),np.linalg.norm(points[:,2]-points[:,1]),np.linalg.norm(points[:,6]-points[:,3]),np.linalg.norm(points[:,4]-points[:,3]),np.linalg.norm(points[:,5]-points[:,4]),np.linalg.norm(points[:,7]-points[:,6]),np.linalg.norm(points[:,8]-points[:,7])])
    theta = JointCarts2Angles(points) #get current human joint angles
    x_hat_temp, x_hat_next_temp = kalman_filter(theta,x_hat) #get the est. and pred. states from the filter
    if not np.isnan(np.sum(x_hat_temp)):
        x_hat = x_hat_temp
        x_hat_next = x_hat_next_temp
    predicted_points = forwardKinematics(x_hat_next[:21],points[:,0],dims)
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
    
    #go from fixed z axis to the pelvis->spine vector
    #align x axis with new z (pelvis->spine vector) crossed with fixed y
    z_1 = vects[:,0]/np.linalg.norm(vects[:,0])
    x_1 = np.cross(np.array([0,1,0]).T,z_1)
    x_1 = x_1/np.linalg.norm(x_1)
    y_1 = np.cross(z_1,x_1)
    y_1 = y_1/np.linalg.norm(y_1)
    R1 = np.concatenate((x_1.reshape((3,1)),y_1.reshape((3,1)),z_1.reshape((3,1))),axis=1)
    theta[0], theta[1], theta[2] = tr2rpy(R1)
    
    #go from spine z axis (z1) to spine->head vector
    #new y axis aligned with new z (spine->head vector) crossed with spine x axis (x1)
    z_2 = vects[:,1]/np.linalg.norm(vects[:,1])
    y_2 = np.cross(z_2,x_1)
    y_2 = y_2/np.linalg.norm(y_2)
    x_2 = np.cross(y_2,z_2)
    x_2 = x_2/np.linalg.norm(x_2)
    R2 = R1.T@np.concatenate((x_2.reshape((3,1)),y_2.reshape((3,1)),z_2.reshape((3,1))),axis=1)
    theta[3], theta[4], theta[5] = tr2rpy(R2)
    
    #go form spine x axis (x_1) to the shoulder->shoulder vector
    #new y axis is spine z axis (z_1) crossed with shoulder->shoulder vector
    x_3 = shoulder2shoulder/np.linalg.norm(shoulder2shoulder)
    y_3 = np.cross(z_1,x_3)
    y_3 = y_3/np.linalg.norm(y_3)
    z_3 = np.cross(x_3,y_3)
    z_3 = z_3/np.linalg.norm(z_3)
    R3 = R1.T@np.concatenate((x_3.reshape((3,1)),y_3.reshape((3,1)),z_3.reshape((3,1))),axis=1)
    theta[6], theta[7], theta[8] = tr2rpy(R3)
   
    R_shoulder = R1@R3 #rotation matrix to go from fixed xyz at pelvis to shoulder xyz
    
    #go from shoulder z axis to the opposite of the shoulder1->elbow1 vector
    #align the elbox y axis with the shoulder->shoulder vector crossed with elbow1->shoulder1
    y_4 = np.cross(shoulder2shoulder,-vects[:,4])
    y_4 = y_4/np.linalg.norm(y_4)
    z_4 = -vects[:,4]
    z_4 = z_4/np.linalg.norm(z_4)
    x_4 = np.cross(y_4,z_4)
    x_4 = x_4/np.linalg.norm(x_4)
    R_e1 = R_shoulder.T@np.concatenate((x_4.reshape((3,1)),y_4.reshape((3,1)),z_4.reshape((3,1))),axis=1)
    theta[9], theta[10], theta[11] = tr2rpy(R_e1)
    
    #align the elbow z axis with the elbow1->wrist1 vector
    #align the wrist y axis with the cross product of the elbow1->wrist1 vector and the shoulder1->elbow1 vector
    y_5 = np.cross(vects[:,5],vects[:,4])
    y_5 = y_5/np.linalg.norm(y_5)
    z_5 = -vects[:,5]/np.linalg.norm(vects[:,5])
    x_5 = np.cross(y_5,z_5)
    x_5 = x_5/np.linalg.norm(x_5)
    R_w1 = R_e1.T@R_shoulder.T@np.concatenate((x_5.reshape((3,1)),y_5.reshape((3,1)),z_5.reshape((3,1))),axis=1)
    theta[12], theta[13], theta[14] = tr2rpy(R_w1)    
    
    #go from shoulder z axis to the opposite of the shoulder2->elbow2 vector
    #align the elbow y axis with the shoulder->shoulder vector crossed with elbow2->shoulder2
    y_6 = np.cross(shoulder2shoulder,-vects[:,6])
    y_6 = y_6/np.linalg.norm(y_6)
    z_6 = -vects[:,6]
    z_6 = z_6/np.linalg.norm(z_6)
    x_6 = np.cross(y_6,z_6)
    x_6 = x_6/np.linalg.norm(x_6)
    R_e2 = R_shoulder.T@np.concatenate((x_6.reshape((3,1)),y_6.reshape((3,1)),z_6.reshape((3,1))),axis=1)
    theta[15], theta[16], theta[17] =  tr2rpy(R_e2)
    
    #align the elbow z axis with the elbow2->wrist2 vector
    #align the wrist y axis with the cross product of the elbow2->wrist2 vector and the shoulder2->elbow2 vector
    y_7 = np.cross(vects[:,7],vects[:,6])
    y_7 = y_7/np.linalg.norm(y_7)
    z_7 = -vects[:,7]/np.linalg.norm(vects[:,7])
    x_7 = np.cross(y_7,z_7)
    x_7 = x_7/np.linalg.norm(x_7)
    R_w2 = R_e2.T@R_shoulder.T@np.concatenate((x_7.reshape((3,1)),y_7.reshape((3,1)),z_7.reshape((3,1))),axis=1)
    theta[18], theta[19], theta[20] =  tr2rpy(R_w2)   
    
    return theta

def wrap2pi(theta):
    over_pi = theta>np.pi
    under_pi = theta<-np.pi
    theta=theta-over_pi*2*np.pi+under_pi*2*np.pi
    return theta

def kalman_filter(theta,x_hat_n):

    error = theta-x_hat_n[:21] #estimation error
    x_hat_np1 = x_hat_n + A*np.tile(error,(3,1)) #predicted state
    #if the estimated state angles are outside of -pi to pi, 
    #then modify them to stay in range
    x_hat_np1[:21]=wrap2pi(x_hat_np1[:21])
    x_hat_n = F@x_hat_np1 #estimated state based on predicted state
    #if the estimated state angles are outside of -pi to pi, 
    #then modify them to stay in range
    x_hat_n[:21]=wrap2pi(x_hat_n[:21])
    
    return x_hat_n, x_hat_np1

def forwardKinematics(theta,pelvis_coords,dimension):
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


def sweepSurfaceHuman(human, human_C, human_RA, human_LA, human_RL, human_LL, Trajectory, grid_spacing):
    #This method invokes all the necessary functions after loading the joint trajectories
    #Determine the area swept out by human. Data stored in XYZ coordinates within patches
    patches_hum = generateSurface(human,Trajectory,grid_spacing)
    patches_shift_C  = generateSurface(human_C,Trajectory,grid_spacing)
    patches_shift_LA = generateSurface(human_LA,Trajectory,grid_spacing)
    patches_shift_RA = generateSurface(human_RA,Trajectory,grid_spacing)    
    legs = False    

    if legs:
        patches_shift_LL = generateSurface(human_LL,Trajectory,grid_spacing)
        patches_shift_RL = generateSurface(human_RL,Trajectory,grid_spacing)

    #Set Boundary conditions for human
    IC = 0
    BC_f_h = boundary_conditions_human(human[-1,:,:], human_C[-1,:,:],human_RA[-1,:,:],human_LA[-1,:,:],human_RL[-1,:,:],human_LL[-1,:,:],Trajectory,grid_spacing,IC, legs)
    
    #Add all components of human sweep together
    if legs:
        human = np.zeros((6,len(patches_hum[0])+len(patches_shift_C[0])+len(patches_shift_LA[0])+len(patches_shift_RA[0])+len(patches_shift_LL[0])+len(patches_shift_RL[0])+len(BC_f_h[0])))
        for ii in range(6):
            human[ii,:] = np.concatenate((patches_hum[ii],patches_shift_C[ii],patches_shift_LA[ii],patches_shift_RA[ii],patches_shift_LL[ii],patches_shift_RL[ii],BC_f_h[ii]), axis=None)
    else:
        human = np.zeros((6,len(patches_hum[0])+len(patches_shift_C[0])+len(patches_shift_LA[0])+len(patches_shift_RA[0])+len(BC_f_h[0])))
        for ii in range(6):
            human[ii,:] = np.concatenate((patches_hum[ii],patches_shift_C[ii],patches_shift_LA[ii],patches_shift_RA[ii],BC_f_h[ii]), axis=None)
      
    return human

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
def getJointParent(node, missingArray):

    #If the node is missing, the parent node should be skipped
    if node in missingArray:
        return 1

    #Node exist but other data may be missing
    if not((node - 1) in missingArray):
        return node - 1
    else:
        return 1
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
def getChildJoint(node, missingArray):
    if not((node + 1) in missingArray):
        return node + 1
    else:
        return getJointParent(node, missingArray)
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
def findVect(U1, U2):
    U12 = np.array([U2[0] - U1[0], U2[1] - U1[1], U2[2] - U1[2]])
    return U12
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
def kalmanTest():
    
    #KALMAN FILTER VISUAL
    #Global Parameters
    grid_spacing = 0.05
    dt = 1/30 #time step
    alpha = 1.0 #gain for change in angle
    beta = 1.0 #gain for change in velocity
    gamma = 0.1 #gain for change in acceleration

    #F, G, and A are matrices for use in the Kalman Filter
    F=np.concatenate((np.concatenate((np.eye(21),dt*np.eye(21),0.5*dt**2*np.eye(21)),axis=1),np.concatenate((np.zeros((21,21)),np.eye(21),dt*np.eye(21)),axis=1),np.concatenate((np.zeros((21,21)),np.zeros((21,21)),np.eye(21)),axis=1)))
    G=np.array([[(1-alpha)*np.eye(21),np.zeros((21,21)),np.zeros((21,21))],[np.zeros((21,21)),(1-beta/dt)*np.eye(21),np.zeros((21,21))],[np.zeros((21,21)),np.zeros((21,21)),(1-2*gamma/dt**2)*np.eye(21)]])
    A = np.concatenate((alpha*np.ones((21,1)), beta/dt*np.ones((21,1)), 2*gamma/dt**2*np.ones((21,1))))
    x_hat = F@np.zeros((63,1)) #initialize the state estimates
    x_hat_next = x_hat #init state predictions
    estimates = x_hat[:21] #store state estimates and predictions
    predictions = x_hat[:21]

    #Testing data
    #-----------------------------------------------------------
    index = 0
    otherway = False
    with open('HumanMotionBad.pickle', 'rb') as handle:
        humanmotion = pickle.load(handle)
    #-----------------------------------------------------------
    fig = plt.figure()
    ax = Axes3D(fig)
    fig1 = plt.figure()
    ax1 = Axes3D(fig1)

    for ii in range(150):
        joints = humanmotion[ii][0]
        missingArray = findMissingPoints(joints)
        #Convert to workspace coordinate system
        joint_location = swapYZ(joints)
        joint_location = transformToFixedFrame(joint_location, -10*math.pi/180)
        #Set any necessary parent joints
        joint_location = setToParents(joint_location, missingArray)
        ax.scatter(joint_location[:,0], joint_location[:,1], joint_location[:,2], marker='.')
        kalPoints = mapSkelToKalman(joint_location)
        kalPoints = kalmanFilter(kalPoints)
        joint_location = mapKalmanToSkel(kalPoints, joint_location)
        ax1.scatter(joint_location[:,0], joint_location[:,1], joint_location[:,2], marker='.')


    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.axes.set_xlim3d(left=-2, right=2) 
    ax.axes.set_zlim3d(bottom=-1, top=3) 
    ax.axes.set_ylim3d(bottom=-2, top=2) 
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.axes.set_xlim3d(left=-2, right=2) 
    ax1.axes.set_zlim3d(bottom=-1, top=3) 
    ax1.axes.set_ylim3d(bottom=-2, top=2) 
    plt.show()
    '''
    fig = plt.figure()
    ax = Axes3D(fig)

    for ii in range(100, 150):
        joint_location = humanmotion[ii][0]
        
        #Find missing data points
        missingArray = findMissingPoints(joint_location)

        #Create nose to allow Kalman filter to use data from back view of human
        comparison = joint_location[0,:] == np.array([0.0, 0.0, 0.0])
        if comparison.all():
            joint_location[0,:], inopperable = headMidPoint(joint_location, missingArray)
            missingArray = np.delete(missingArray,[0])
            if inopperable:
                print('Head Fail')
                continue
        
        #Convert to workspace coordinate system
        joint_location = swapYZ(joint_location)
        joint_location = transformToFixedFrame(joint_location, -10*math.pi/180)
        
        #Alter data Indices then pass to Kalman Filter, then alter back
        kalPoints = mapSkelToKalman(joint_location)
        kalPoints = kalmanFilter(kalPoints)
        joint_location = mapKalmanToSkel(kalPoints, joint_location)
        
        
        ax.scatter(joint_location[:,0], joint_location[:,1], joint_location[:,2], marker='.')
        
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()
    '''
def findUnitVect(U1, U2):
    U12 = np.array([U2[0] - U1[0], U2[1] - U1[1], U2[2] - U1[2]])
    U12 = U12/np.linalg.norm(U12)

    if np.linalg.norm(U12) < 0.001:
        divByZero = True
    else:
        divByZero = False

    return U12, divByZero
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
def transformMat(Ux, Uy, Uz, P):
    #Find rotation between fixed basis and torso basis then translate to point P
    T_ba = np.zeros((4,4))
    T_ba[0:3,0] = Ux
    T_ba[0:3,1] = Uy
    T_ba[0:3,2] = Uz
    T_ba[3,3] = 1
    T_ba[0:3,3] = P

    return T_ba
def checkDims():
     #Load initial robot plans: These are in format that ROS will output
    with open('dim_data.pickle', 'rb') as handle:
        dims = pickle.load(handle)
    for ii in range(len(dims[0])):
        plt.plot(range(len(dims)),dims[:,ii],)
    plt.show()
    print(dims.mean(axis=0))
    print(dims.std(axis=0)) 
def getChildShoulder(node, missingArray):
    if not((node + 1) in missingArray):
        return node + 1
    elif not((node + 2) in missingArray):
        return node + 2
    else:
        return getJointParent(node, missingArray)
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
def filterOutliers(joint_location, missingArray):
    inopperable = False

    #Abort this frame if both shoulders are gone
    if (2 in missingArray) and (5 in missingArray):
        inopperable = True
        return inopperable
    
    #Eliminate noisy frames from consideration by making sure problem points are too far from each other
    #Problem points include: Feet, Shoulders, hands
    nom_shoulder =  0.2
    nom_arm = 0.3
    nom_forearm = 0.30525
    nom_neck = 0.1974
    nom_thigh = 0.4165
    nom_shin = 0.4562
    err_shoulder = 0.3
    err_arm = 0.25
    err_forearm = 0.35
    err_neck = 0.15
    err_thigh = 0.25
    err_shin = 0.35
    '''
    #Example of too picky of an algorithm
    err_shoulder = 0.08
    err_arm = 0.08
    err_forearm = 0.08
    err_neck = 0.08
    err_thigh = 0.08
    err_shin = 0.08
    '''
    '''
    #Algorithm not picky enough
    err_shoulder = 1
    err_arm = 1
    err_forearm = 1
    err_neck = 1
    err_thigh = 1
    err_shin = 1
    '''

    #check left and right shoulders
    if abs(np.linalg.norm(findVect(joint_location[1,:],joint_location[2,:]))-nom_shoulder) > err_shoulder or abs(np.linalg.norm(findVect(joint_location[1,:],joint_location[5,:]))-nom_shoulder) > err_shoulder:
        inopperable = True

    #check neck
    if abs(np.linalg.norm(findVect(joint_location[1,:],joint_location[0,:]))-nom_neck) > err_neck:
        inopperable = True

    #Check hand distance from elbows
    if abs(np.linalg.norm(findVect(joint_location[3,:],joint_location[4,:]))-nom_forearm) > err_forearm or abs(np.linalg.norm(findVect(joint_location[6,:],joint_location[7,:]))-nom_forearm) > err_forearm:
        #Only check if the joints have been detected
        if not(3 in missingArray) and not(4 in missingArray) and not(5 in missingArray) and not(6 in missingArray):
            inopperable = True
            
    #Check elbow distance to shoulder
    if abs(np.linalg.norm(findVect(joint_location[3,:],joint_location[2,:]))-nom_arm) > err_arm or abs(np.linalg.norm(findVect(joint_location[6,:],joint_location[5,:]))-nom_arm) > err_arm:
        #Only check if the joints have been detected
        if not(2 in missingArray) and not(3 in missingArray) and not(5 in missingArray) and not(6 in missingArray):
            inopperable = True
    if legs:   
        #Check feet distance to knee
        if abs(np.linalg.norm(findVect(joint_location[9,:],joint_location[10,:]))-nom_shin) > err_shin or abs(np.linalg.norm(findVect(joint_location[12,:],joint_location[13,:]))-nom_shin) > err_shin:
            #Only check if the joints have been detected
            if not(9 in missingArray) and not(10 in missingArray) and not(12 in missingArray) and not(13 in missingArray):
                inopperable = True    
        #Check knee distance to hip
        if abs(np.linalg.norm(findVect(joint_location[8,:],joint_location[9,:]))-nom_thigh) > err_thigh or abs(np.linalg.norm(findVect(joint_location[11,:],joint_location[12,:]))-nom_thigh) > err_thigh:
            #Only check if the joints have been detected
            if not(8 in missingArray) and not(9 in missingArray) and not(11 in missingArray) and not(12 in missingArray):
                inopperable = True

    #If the torso or hips are not detected, this frame is inopperable
    requiredPoints = np.array([1,8,11])
    for ii in requiredPoints:
        if ii in missingArray:
            inopperable = True

    return inopperable

def skelToBoundPlot():

    with open('HumanMotionGood.pickle', 'rb') as handle:
                humanmotion = pickle.load(handle)
    #0-10 28 57 61 62 64 65
    #fig = plt.figure()
    #ax = Axes3D(fig)
    inds = np.array([3, 28, 57])
    #for ii in range(80,len(humanmotion),1):
    for ii in inds:
        fig = plt.figure()
        ax = Axes3D(fig)
        print(ii)
        
        joint_location = humanmotion[ii]
        joint_location = joint_location[0,:,:]
        missingArray = findMissingPoints(joint_location)
        joint_location = swapYZ(joint_location)
        point_cloud = transformToFixedFrame(joint_location, -10*math.pi/180)
        point_cloud[0,:], inopperable = headMidPoint(point_cloud, missingArray)
        #Child Parent Structure
        joint_location = setToParents(joint_location, missingArray)
        #Filter Outliers
        if filterOutliers(joint_location, missingArray):
            print('Filtered')

        
        #Plot for skeleton
        ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2],s=40, c='b', marker='.')
        a = np.array([1,1,1,1,1,2,3,5,6,8,9,11,12])
        b = np.array([0,8,11,2,5,3,4,6,7,9,10,12,13])
        for jj in range(len(a)):
            ax.plot([point_cloud[a[jj],0], point_cloud[b[jj],0]],[point_cloud[a[jj],1], point_cloud[b[jj],1]],[point_cloud[a[jj],2], point_cloud[b[jj],2]], linewidth=2, color='blue')
        
        #Plot for boundary curves
        boundary_points, boundary_points_C, boundary_points_RA, boundary_points_LA, boundary_points_RL, boundary_points_LL, success = boundaryPoints(joint_location, missingArray)
        ax.plot(boundary_points[0,:],boundary_points[1,:],boundary_points[2,:], linewidth=2, color='red')
        #ax.scatter(boundary_points[0,:],boundary_points[1,:],boundary_points[2,:], linewidth=2, s=40, c='r', marker='.')
        ax.plot([boundary_points[0,0],boundary_points[0,-1]],[boundary_points[1,0],boundary_points[1,-1]],[boundary_points[2,0],boundary_points[2,-1]], linewidth=1, color='red')     
        '''
        
        #Plot Orthoganol Boundary Curves
        ax.plot(boundary_points_C[0,:],boundary_points_C[1,:],boundary_points_C[2,:], linewidth=2, color='green')
        ax.plot([boundary_points_C[0,0],boundary_points_C[0,-1]],[boundary_points_C[1,0],boundary_points_C[1,-1]],[boundary_points_C[2,0],boundary_points_C[2,-1]], linewidth=1, color='green')     
        ax.plot(boundary_points_LA[0,:],boundary_points_LA[1,:],boundary_points_LA[2,:], linewidth=2, color='green')
        ax.plot(boundary_points_RA[0,:],boundary_points_RA[1,:],boundary_points_RA[2,:], linewidth=2, color='green')
        ax.plot(boundary_points_LL[0,:],boundary_points_LL[1,:],boundary_points_LL[2,:], linewidth=2, color='green')
        ax.plot(boundary_points_RL[0,:],boundary_points_RL[1,:],boundary_points_RL[2,:], linewidth=2, color='green')
        '''
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.axes.set_xlim3d(left=-1, right=1) 
        ax.axes.set_zlim3d(bottom=0, top=2) 
        ax.axes.set_ylim3d(bottom=-1, top=1) 
        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.show()

def BPToSurface():
    human = np.load('Total.npy')
    human_C = np.load('Total_C.npy')
    human_LA = np.load('Total_LA.npy')
    human_RA = np.load('Total_RA.npy')
    human_LL = np.load('Total_LL.npy')
    human_RL = np.load('Total_RL.npy')
    Traj_hum = np.zeros((2,len(human)))
    Traj_hum[0,:] = np.linspace(0,5,len(human))
    Traj_hum[1,:] = np.ones(len(human))
    grid_spacing = 0.08
    human_pc = sweepSurfaceHuman(human[0::10,:,:], human_C[0::10,:,:], human_RA[0::10,:,:], human_LA[0::10,:,:], human_RL[0::10,:,:], human_LL[0::10,:,:], Traj_hum[:,0::10], grid_spacing)
    visual(human_pc, np.array([]))

def savedToBoundPlot():

    boundary_points_total = np.load('Total.npy')
    boundary_points_total_C = np.load('Total_C.npy')
    boundary_points_total_LA = np.load('Total_LA.npy')
    boundary_points_total_RA = np.load('Total_RA.npy')
    boundary_points_total_LL = np.load('Total_LL.npy')
    boundary_points_total_RL = np.load('Total_RL.npy')
    #111
    
    indpoints = [0,15,30]
    for ii in indpoints:
        fig = plt.figure()
        ax = Axes3D(fig)
        #Plot for boundary curves
        boundary_points = boundary_points_total[ii]
        boundary_points_C = boundary_points_total_C[ii]
        boundary_points_RA = boundary_points_total_RA[ii]
        boundary_points_LA = boundary_points_total_LA[ii]
        boundary_points_RL = boundary_points_total_RL[ii]
        boundary_points_LL = boundary_points_total_LL[ii]
        ax.plot(boundary_points[0,:],boundary_points[1,:],boundary_points[2,:], linewidth=2, color='red')
        #ax.scatter(boundary_points[0,:],boundary_points[1,:],boundary_points[2,:], linewidth=2, s=40, c='r', marker='.')
        ax.plot([boundary_points[0,0],boundary_points[0,-1]],[boundary_points[1,0],boundary_points[1,-1]],[boundary_points[2,0],boundary_points[2,-1]], linewidth=2, color='red')     
        '''
        #Plot Orthoganol Boundary Curves
        ax.plot(boundary_points_C[0,:],boundary_points_C[1,:],boundary_points_C[2,:], linewidth=2, color='blue')
        ax.plot([boundary_points_C[0,0],boundary_points_C[0,-1]],[boundary_points_C[1,0],boundary_points_C[1,-1]],[boundary_points_C[2,0],boundary_points_C[2,-1]], linewidth=1, color='blue')     
        ax.plot(boundary_points_LA[0,:],boundary_points_LA[1,:],boundary_points_LA[2,:], linewidth=2, color='blue')
        ax.plot(boundary_points_RA[0,:],boundary_points_RA[1,:],boundary_points_RA[2,:], linewidth=2, color='blue')
        ax.plot(boundary_points_LL[0,:],boundary_points_LL[1,:],boundary_points_LL[2,:], linewidth=2, color='blue')
        ax.plot(boundary_points_RL[0,:],boundary_points_RL[1,:],boundary_points_RL[2,:], linewidth=2, color='blue')
        '''
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.axes.set_xlim3d(left=-1, right=1) 
        ax.axes.set_zlim3d(bottom=0, top=2) 
        ax.axes.set_ylim3d(bottom=-1, top=1) 
        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.show()
def dimFilter(missingArray):
    inopperable = False
    needed = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    for ii in range(len(needed)):
        point = needed[ii]
        if (point in missingArray):
            inopperable = True
            return inopperable
    
    return inopperable

def getDims():
    with open('HumanMotionBad.pickle', 'rb') as handle:
        humanmotion = pickle.load(handle)
    Dims = np.zeros([11,151])
    ind = 0
    skip = 0
    for ii in range(0,len(humanmotion)):
        joint_location = humanmotion[ii]
        joint_location = joint_location[0,:,:]
        missingArray = findMissingPoints(joint_location)
        joint_location = swapYZ(joint_location)
        point_cloud = transformToFixedFrame(joint_location, -10*math.pi/180)
        point_cloud[0,:], inopperable1 = headMidPoint(point_cloud, missingArray)
        joint_location = setToParents(joint_location, missingArray)
        inopperable2 = dimFilter(point_cloud)
        if inopperable1 or inopperable2:
            print('skipped')
            skip = skip + 1
            continue

        p1_vect = [13, 5, 6, 7, 2, 12, 4, 9, 10, 3, 0]
        p2_vect = [12, 1, 5, 6, 1, 11, 3, 8, 9, 2, 1]
        names = ['Knee/Ankle (Right)','Torso/Shoulder (Left)','Shoulder/Elbow (Left)','Elbow/Wrist (Left)','Torso/Shoulder (Right)','Hip/Knee (Right)','Elbow/Wrist (Right)','Hip/Knee (Left)','Knee/Ankle (Left)','Shoulder/Elbow (Right)','Head/Torso']
        count = 0
        for jj in range(11):
            p1 = p1_vect[count]
            p2 = p2_vect[count]
            Dims[jj,ind] = np.linalg.norm(point_cloud[p1,:]-point_cloud[p2,:])
            count = count + 1
        ind = ind + 1
    endL = len(humanmotion) - skip
    '''
    fig, axs = plt.subplots(1,6)
    for kk in range(0,6):
        axs[kk].hist(Dims[kk,0:endL], bins=30)
        av = np.mean(Dims[kk,0:endL])
        stdev = np.std(Dims[kk,0:endL])
        axs[kk].title.set_text(names[kk] + '\nMean: ' + str(round(av,4)) + '\nStandard Deviation: '+str(round(stdev,4)))
        axs[kk].set_xlabel('Length (m)')
        axs[kk].set_xlim([0.15,1])
        axs[kk].set_ylim([0,60])
    '''
    fig1, axs1 = plt.subplots(1,5)  
    step = 0  
    for kk in range(6,11):
        axs1[step].hist(Dims[kk,0:endL], bins=30)
        av = np.mean(Dims[kk,0:endL])
        stdev = np.std(Dims[kk,0:endL])
        axs1[step].title.set_text(names[kk] + '\nMean: ' + str(round(av,4)) + '\nStandard Deviation: '+str(round(stdev,4)))
        axs1[step].set_xlabel('Length (m)')
        axs1[step].set_xlim([0,1.5])
        step = step + 1

    plt.show()
    
def MA(x):
    return np.convolve(x, np.ones(30), 'valid') / 30

def timeEval():

    #for ii in range(len(env_sweeps)):
    #    visual(env_sweeps[ii][0], env_sweeps[ii][1])

    with open('SkelTime_3_time.pickle', 'rb') as handle:
        SkelTime = pickle.load(handle)

    with open('SweepTime_3_time.pickle', 'rb') as handle:
        SweepTime = pickle.load(handle)

    with open('ColTime_3_time.pickle', 'rb') as handle:
        ColTime = pickle.load(handle)

    with open('BoundTime_3_time.pickle', 'rb') as handle:
        BoundTime = pickle.load(handle)

    with open('TotalTime_3_time.pickle', 'rb') as handle:
        TotalTime = pickle.load(handle)
    
    with open('FramesRejected_3_time.pickle', 'rb') as handle:
        FramesRejected = pickle.load(handle)


    iters = range(1,len(FramesRejected)+1)
    iters_skel = range(1,len(SkelTime)+1)
    '''
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    fig.tight_layout(pad=1.0)

    ax1.plot(iters, 1000*SweepTime)
    ax1.set_ylabel('Run Time (ms)')
    ax1.set_title('Time Data For Surface Sweep')

    ax2.plot(iters, 1000*KalTime)
    ax2.set_title('Time Data For Kalman Filter')
    ax2.set_ylabel('Run Time (ms)')

    ax3.plot(iters, 1000*ColTime)
    ax3.set_title('Time Data For Collision Check')
    ax3.set_ylabel('Run Time (ms)')
    ax3.set_xlabel('Iteration')

    ax4.plot(iters, 1000*BoundTime)
    ax4.set_title('Time Data For Boundary Curve Calculation')
    ax4.set_ylabel('Run Time (ms)')
    '''
    print('Average Total Time: ')
    print(np.mean(TotalTime)*1000)
    TT = np.mean(TotalTime)*1000
    print('Average Skeleton Tracking Time (ms): ')
    print(np.mean(SkelTime)*1000)
    SK = np.mean(SkelTime)*1000
    print('Average Boundary Curve Time (ms): ')
    print(np.mean(BoundTime)*1000)
    BT = np.mean(BoundTime)*1000
    print('Average Surface Sweep Time (ms): ')
    print(np.mean(SweepTime)*1000)
    ST = np.mean(SweepTime)*1000
    print('Average Collision Check Time (ms): ')
    print(np.mean(ColTime)*1000)
    CT = np.mean(ColTime)*1000
    first = ' '.join((r'$Average$', r'$Surface$', r'$Sweep$', r'$Tracking$', r'$Time=%.2f$ ms' % (ST, )))
    second = ' '.join((r'$Average$', r'$Collision$', r'$Check$', r'$Time=%.2f$ ms' % (CT, )))
    third = ' '.join((r'$Average$', r'$Boundary$', r'$Curve$', r'$Time=%.2f$ ms' % (BT, )))

    textstr = '\n'.join((first,second,third))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    font = {'family' : 'normal',
            'size'   : 20}

    plt.rc('font', **font)
    ax1 = plt.figure()
    ax1.text(0.2, 0.75, textstr, fontsize=20, bbox=props)
    iters_MA = iters[:-29]
    plt.plot(iters_MA, 1000*MA(SweepTime), 'r-',label='Surface Sweep',zorder=10)
    plt.plot(iters_MA, 1000*MA(ColTime), 'b-*',label='Collision Check',zorder=10)
    plt.plot(iters_MA, 1000*MA(BoundTime),'g--', label='Boundary Curve Calculation',zorder=10)
    plt.plot(iters, 1000*(SweepTime), 'r-', alpha=0.2,zorder=1)
    plt.plot(iters, 1000*(ColTime), 'b-*', alpha=0.2,zorder=1)
    plt.plot(iters, 1000*(BoundTime),'g--', alpha=0.2,zorder=1)
    plt.ylabel('Run Time (ms)')
    plt.xlabel('Iteration')
    plt.title('Total Time Data and Submodules Time Data')
    plt.legend()
    '''
    ax2.plot(iters, 1000*SweepTime, label='Surface Sweep','b')
    ax2.plot(iters, 1000*ColTime, label='Collision Check','g')
    ax2.plot(iters, 1000*BoundTime, label='Boundary Curve Calculation','m')
    ax2.plot(iters, 1000*TotalTime, '--', label='Total Run Time')
    ax2.set_title('Total Time Data')
    ax2.set_ylabel('Run Time (ms)')
    ax2.legend()
    '''
    
    fig3,ax3 = plt.subplots()
    ax = ax3.twinx()
    ax.plot(iters, FramesRejected,label='Frames Skipped',zorder=-10)
    ax3.plot(iters, 1000*TotalTime, 'r--', label='Total Run Time')
    ax3.set_ylabel('Total Run Time (ms)')
    ax3.set_xlabel('Iteration')
    ax.set_ylabel('Frames Skipped')
    ax.set_ylim([0, 15000])
    ax3.set_title('Number of Frames Skipped Each Time the Skeleton Tracker was Called')
    ax3.legend(loc='upper center')
    ax.legend(loc='upper right')
    
    
    '''
    #Plot skel tracking data from other thread
    fig2 = plt.figure()
    plt.plot(iters_skel[1:], 1000*SkelTime[1:], label='Skeleton Tracking Check')
    plt.ylabel('Run Time (ms)')
    plt.xlabel('Iteration')
    plt.title('Skeleton Tracking Thread Run Time')
    '''
    plt.show()
def visual(point_cloud, collisionPoints):
    colmap = plt.get_cmap('jet')
    minn, maxx = point_cloud[3,:].min(), point_cloud[3,:].max()
    norm = cm.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap=colmap)
    fig = plt.figure()
    ax = Axes3D(fig)
    #Plot
    if collisionPoints.size > 0:
        #Plot sweeps with transparency
        ax.scatter(point_cloud[0,:], point_cloud[1,:], point_cloud[2,:], c=m.to_rgba(point_cloud[3,:]),  marker='.',zorder=1)
        ax.scatter(collisionPoints[:,0], collisionPoints[:,1], collisionPoints[:,2], c='red', s=500, marker='.',zorder=20)
    else:
        #Plot sweeps with no transparency
        ax.scatter(point_cloud[0,:], point_cloud[1,:], point_cloud[2,:], c=m.to_rgba(point_cloud[3,:]), marker='.')
    
    m.set_array(point_cloud[3,:])
    fig.colorbar(m,label='Time')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.axis('off')
    ax.axes.set_xlim3d(left=-1, right=1) 
    ax.axes.set_zlim3d(bottom=0, top=2) 
    ax.axes.set_ylim3d(bottom=-2, top=0) 
    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
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

def robSweep():
    grid_spacing = 0.05
    #Load initial robot plans: These are in format that ROS will output
    with open('robotSchedule_sweep.pickle', 'rb') as handle:
        robotSchedule = pickle.load(handle)
    robotScheduleSweep = defaultdict(list)
    robotScheduleTime = defaultdict(list)

    #Conduct the Offline Trajectory Sweeps for Robot
    for ii in range(int(len(robotSchedule))):
        jointTrajectory = robotSchedule[ii+1][0]
        Trajectory = robotSchedule[ii+1][1]
        robotScheduleSweep[ii] = sweepSurfaceRobot(jointTrajectory, Trajectory, grid_spacing)
        robotScheduleTime[ii] = robotSchedule[ii+1][1][0][-1]

    for ii in range(len(robotScheduleSweep)):
        print(robotScheduleSweep[ii][0:4])
        visual(robotScheduleSweep[ii][0:4], np.array([]))

legs = True
#BPToSurface()
#savedToBoundPlot()
skelToBoundPlot()
#timeEval()
#robSweep()
#checkDims()
#getDims()
#with open('rob.pickle', 'rb') as handle:
#        rob_pc = pickle.load(handle)

#print(rob_pc)
#print(np.shape(rob_pc))
'''
with open('col_sweeps_3.pickle', 'rb') as handle:
        col_sweeps = pickle.load(handle)
with open('env_sweeps_3.pickle', 'rb') as handle:
        env_sweeps = pickle.load(handle)
for ii in range(len(env_sweeps)):
    visual(env_sweeps[ii][0], col_sweeps[ii][0])
'''