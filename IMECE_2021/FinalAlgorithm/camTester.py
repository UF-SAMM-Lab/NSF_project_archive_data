#!/usr/bin/env my_pymc_env
import util as cmr
import cv2
import time
import pyrealsense2 as rs
import numpy as np
from skeletontracker import skeletontracker
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from testing_data import *
from kinematics import *
from generate_surface import *
from collision_check import collisionCheck
import cProfile
from collections import defaultdict

def render_ids_3d(render_image, skeletons_2d, depth_map, depth_intrinsic, joint_confidence):
    thickness = 1
    text_color = (255, 255, 255)
    rows, cols, channel = render_image.shape[:3]
    distance_kernel_size = 5
    # calculate 3D keypoints and display them
    joint_location = np.zeros([len(skeletons_2d),len(skeletons_2d[0].joints),3])
    for skeleton_index in range(len(skeletons_2d)):
        skeleton_2D = skeletons_2d[skeleton_index]
        joints_2D = skeleton_2D.joints
        did_once = False
        for joint_index in range(len(joints_2D)):
            if did_once == False:
                cv2.putText(
                    render_image,
                    "id: " + str(skeleton_2D.id),
                    (int(joints_2D[joint_index].x), int(joints_2D[joint_index].y - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    text_color,
                    thickness,
                )
                did_once = True
            # check if the joint was detected and has valid coordinate
            if skeleton_2D.confidences[joint_index] > joint_confidence:
                distance_in_kernel = []
                low_bound_x = max(
                    0,
                    int(
                        joints_2D[joint_index].x - math.floor(distance_kernel_size / 2)
                    ),
                )
                upper_bound_x = min(
                    cols - 1,
                    int(joints_2D[joint_index].x + math.ceil(distance_kernel_size / 2)),
                )
                low_bound_y = max(
                    0,
                    int(
                        joints_2D[joint_index].y - math.floor(distance_kernel_size / 2)
                    ),
                )
                upper_bound_y = min(
                    rows - 1,
                    int(joints_2D[joint_index].y + math.ceil(distance_kernel_size / 2)),
                )
                for x in range(low_bound_x, upper_bound_x):
                    for y in range(low_bound_y, upper_bound_y):
                        distance_in_kernel.append(depth_map.get_distance(x, y))
                median_distance = np.percentile(np.array(distance_in_kernel), 50)
                depth_pixel = [
                    int(joints_2D[joint_index].x),
                    int(joints_2D[joint_index].y),
                ]
                if median_distance >= 0.3:
                    point_3d = rs.rs2_deproject_pixel_to_point(
                        depth_intrinsic, depth_pixel, median_distance
                    )
                    joint_location[skeleton_index, joint_index,:] = point_3d
                    point_3d = np.round([float(i) for i in point_3d], 3)
                    point_str = [str(x) for x in point_3d]
                    cv2.putText(
                        render_image,
                        str(point_3d),
                        (int(joints_2D[joint_index].x), int(joints_2D[joint_index].y)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.4,
                        text_color,
                        thickness,
                    )

    return joint_location

# Main content begins
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

    #Find rotate abound x to align with fixed
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



def headPoints(boundary_points, boundary_points_C, joint_location, missingArray):

    inopperable =  False
    divByZero1 = False
    divByZero2 = False
    divByZero = False

    error_head = 0.55

    #CASE 1: Both ear points are detected
    if not (16 in missingArray or 17 in missingArray):
        Uhoriz, divByZero1 = findUnitVect(joint_location[16,:],joint_location[17,:])
        #Create vector from center of head to nose
        mid_point = np.array([(joint_location[16,0] + joint_location[17,0])/2, (joint_location[16,1] + joint_location[17,1])/2, (joint_location[16,2] + joint_location[17,2])/2])

        #Check if the head point is reasonable or if it is shooting off to singularity 
        if abs(np.linalg.norm(findVect(joint_location[1,:],mid_point))) > error_head:
            inopperable = True
        else:
            Hvect, divByZero2 = findUnitVect(mid_point, joint_location[1,:])
            norm = np.cross(Hvect, Uhoriz)
            Uvert = np.cross(norm,Uhoriz)
            boundary_points[:, 8] = joint_location[16,:] + Uvert * 0.15
            boundary_points[:, 9] = joint_location[17,:] + Uvert * 0.15
            
            #Find head width and assign points to core ortho curve
            width = np.linalg.norm(findVect( boundary_points[:, 8],  boundary_points[:, 9] ))/2
            boundary_points_C[:, 2] = mid_point + Uvert * 0.15 + norm * width
            boundary_points_C[:, 3] = mid_point + Uvert * 0.15 - norm * width


    #CASE 2: Previous case invalid: Use left ear and nose 
    elif not (17 in missingArray or 0 in missingArray):

        #Check if the head point is reasonable or if it is shooting off to singularity 
        if abs(np.linalg.norm(findVect(joint_location[1,:],joint_location[0,:]))) > error_head:
            inopperable = True
        else:
            Uhoriz, divByZero1 = findUnitVect(joint_location[0,:],joint_location[17,:])
            #Create vector from center of head to nose
            Hvect, divByZero2 = findUnitVect(joint_location[0,:], joint_location[1,:])
            norm = np.cross(Hvect, Uhoriz)
            Uvert = np.cross(norm,Uhoriz)
            boundary_points[:, 8] = joint_location[0,:] + Uvert * 0.15 - Uhoriz * 0.1
            boundary_points[:, 9] = joint_location[17,:] + Uvert * 0.15

            #Find head width and assign points to core ortho curve
            width = np.linalg.norm(findVect( boundary_points[:, 8],  boundary_points[:, 9] ))/2
            boundary_points_C[:, 2] = joint_location[0,:] + Uvert * 0.15 + norm * width
            boundary_points_C[:, 3] = joint_location[0,:] + Uvert * 0.15 - norm * width


    #CASE 3: Previous case invalid: Use right ear and nose
    elif not (16 in missingArray or 0 in missingArray):

        #Check if the head point is reasonable or if it is shooting off to singularity 
        if abs(np.linalg.norm(findVect(joint_location[1,:],joint_location[0,:]))) > error_head:
            inopperable = True
        else:
            Uhoriz, divByZero1 = findUnitVect(joint_location[16,:],joint_location[0,:])
            #Create vector from center of head to nose
            Hvect, divByZero2 = findUnitVect(joint_location[0,:], joint_location[1,:])
            norm = np.cross(Hvect, Uhoriz)
            Uvert = np.cross(norm,Uhoriz)
            boundary_points[:, 8] = joint_location[16,:] + Uvert * 0.15
            boundary_points[:, 9] = joint_location[0,:] + Uvert * 0.15 + Uhoriz * 0.1

            #Find head width and assign points to core ortho curve
            width = np.linalg.norm(findVect( boundary_points[:, 8],  boundary_points[:, 9] ))/2
            boundary_points_C[:, 2] = joint_location[0,:] + Uvert * 0.15 + norm * width
            boundary_points_C[:, 3] = joint_location[0,:] + Uvert * 0.15 - norm * width


    #CASE 4: Previous case invalid: Use 2 eyes
    elif not (14 in missingArray or 15 in missingArray):

        Uhoriz, divByZero1 = findUnitVect(joint_location[14,:],joint_location[15,:])
        #Create vector from center of head to nose
        mid_point = np.array([(joint_location[14,0] + joint_location[15,0])/2, (joint_location[14,1] + joint_location[15,1])/2, (joint_location[14,2] + joint_location[15,2])/2])

        #Check if the head point is reasonable or if it is shooting off to singularity 
        if abs(np.linalg.norm(findVect(joint_location[1,:],mid_point))) > error_head:
            inopperable = True
        else:
                
            Hvect, divByZero2 = findUnitVect(mid_point, joint_location[1,:])
            norm = np.cross(Hvect, Uhoriz)
            Uvert = np.cross(norm,Uhoriz)
            boundary_points[:, 8] = joint_location[14,:] + Uvert * 0.13 - Uhoriz * 0.03
            boundary_points[:, 9] = joint_location[15,:] + Uvert * 0.13 + Uhoriz * 0.03

            #Find head width and assign points to core ortho curve
            width = np.linalg.norm(findVect( boundary_points[:, 8],  boundary_points[:, 9] ))/2
            boundary_points_C[:, 2] = mid_point + Uvert * 0.15 + norm * width
            boundary_points_C[:, 3] = mid_point + Uvert * 0.15 - norm * width

        
    #CASE 4: Not enough points to opperate
    else:
        inopperable = True

    #Perform a few checks to filter out singularities
    #Check for any caught singularities
    if divByZero1 or divByZero2:
        divByZero = True
    #Check calculated points
    if abs(np.linalg.norm(findVect(joint_location[1,:],boundary_points[:,8]))) > 0.9 or abs(np.linalg.norm(findVect(joint_location[1,:],boundary_points[:,9]))) > 0.9:
        inopperable = True

    return boundary_points, boundary_points_C, inopperable, divByZero

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
    #check left and right shoulders
    if abs(np.linalg.norm(findVect(joint_location[1,:],joint_location[2,:]))) > error_shoulder or abs(np.linalg.norm(findVect(joint_location[1,:],joint_location[5,:]))) > error_shoulder:
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

def boundaryPoints(joint_location):

    #Find missing data points
    missingArray = findMissingPoints(joint_location)
    #Set any necessary parent joints
    joint_location = setToParents(joint_location, missingArray)

    #Check if shoulders are ok (they are usually poorly detected)
    #If one is bad but the other us good, use the good to fix the bad
    if (2 in missingArray) and not(5 in missingArray):
        r5, divByZero =findUnitVect(joint_location[1,:], joint_location[5,:])
        joint_location[2,:] = joint_location[1,:] - r5
    if (5 in missingArray) and not(2 in missingArray):
        r4, divByZero =findUnitVect(joint_location[1,:], joint_location[2,:])
        joint_location[5,:] = joint_location[1,:] - r4

    #Initialize boundary points to max boundary. Also initialize ortho curves for arms, legs, and core
    boundary_points_RA = np.zeros((3,6))
    boundary_points_LA = np.zeros((3,6))
    boundary_points_RL = np.zeros((3,6))
    boundary_points_LL = np.zeros((3,6))
    boundary_points_C  = np.zeros((3,6))
    boundary_points    = np.zeros((3,27))

    #Boundary points of Head (boundary points contains filter for head points for conveinence)
    boundary_points, boundary_points_C, inopperable, divByZero = headPoints(boundary_points, boundary_points_C, joint_location, missingArray)

    #Perform a series of checks to asses if the frame's data is of good quality
    inopperable = filterOutliers(joint_location, missingArray)

    #Deterine the general orientation of the skeleton by defining a plane between the two hip points and the torso point
    #Find two vectors with the three points to define the plane
    Uah, divByZero = findUnitVect(joint_location[8,:], joint_location[1,:])
    Uak, divByZero = findUnitVect(joint_location[11,:], joint_location[1,:])

    #Use cross product to get norm
    if abs(np.dot(Uah, Uak) - 1) < 0.00001:
        inopperable = True
    else:
        norm = np.cross(Uah, Uak)/np.linalg.norm(np.cross(Uah, Uak))
    
    if np.linalg.norm(np.cross(Uah, Uak)) < 0.0001:
        inopperable = True

    if divByZero:
        inopperable = True

    #If bare minimum points are present, procede
    success = True
    if(not inopperable):
                
        #Find mid point of waist
        mid_point = np.array([(joint_location[8,0] + joint_location[11,0])/2, (joint_location[8,1] + joint_location[11,1])/2, (joint_location[8,2] + joint_location[11,2])/2])


        #Waist points are kept as is
        boundary_points[:, 0] = joint_location[8,:]
        boundary_points[:, 17] = joint_location[11,:]


        #Boundary points of torso 
        boundary_points, boundary_points_C, divByZero = torsoTransform(mid_point, norm, boundary_points, boundary_points_C, joint_location[1,:])
        

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

    else:
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

#Function to plot swept surfaces
def visual(patches, collisionPoints):
    colmap = plt.get_cmap('jet')
    minn, maxx = patches[3,:].min(), patches[3,:].max()
    norm = cm.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap=colmap)
    fig = plt.figure()
    ax = Axes3D(fig)
    #Plot sweeps with transparency
    ax.scatter(patches[0,:], patches[1,:], patches[2,:], c=m.to_rgba(patches[3,:]), alpha=0.1)
    #Plot sweeps with no transparency
    ax.scatter(patches[0,:], patches[1,:], patches[2,:], c=m.to_rgba(patches[3,:]))
    #Plot collision points
    ax.scatter(collisionPoints[:,0], collisionPoints[:,1], collisionPoints[:,2], c='red', s=100)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.axes.set_xlim3d(left=-1, right=1) 
    ax.axes.set_zlim3d(bottom=0, top=2) 
    ax.axes.set_ylim3d(bottom=-1, top=1) 
    m.set_array(patches[3,:])
    plt.show()

#This method invokes all the necessary functions after loading the joint trajectories
def SurfaceSweep(human, human_C, human_LA, human_LL, human_RA, human_RL, Trajectory, jointTraj_rob, Traj_rob, grid_spacing):
    
    #Determine XYZ position of each joint at each pose for robot
    a = 12
    b = len(jointTraj_rob[0])
    points = np.zeros(shape=(b,4,a))
    points_shift = np.zeros(shape=(b,4,a))

    for ii in range(0,len(jointTraj_rob[0])):
        points[ii,:,:] = JointPoints(jointTraj_rob[:,ii], a)
        points_shift[ii,:,:] = JointPoints_shift(jointTraj_rob[:,ii], a)
   
    #Determine the area swept out by human. Data stored in XYZ coordinates within patches
    patches_hum = generateSurface(human,Trajectory,grid_spacing)
    patches_shift_C  = generateSurface(human_C,Trajectory,grid_spacing)
    patches_shift_LA = generateSurface(human_LA,Trajectory,grid_spacing)
    patches_shift_RA = generateSurface(human_RA,Trajectory,grid_spacing)
    patches_shift_LL = generateSurface(human_LL,Trajectory,grid_spacing)
    patches_shift_RL = generateSurface(human_RL,Trajectory,grid_spacing)

    
    #Set Boundary conditions for human
    #Set initial conditions
    IC = 1
    BC_i_h = boundary_conditions_human(human[0,:,:], human_C[0,:,:],human_LA[0,:,:],human_RA[0,:,:],human_LL[0,:,:],human_RL[0,:,:],Trajectory,grid_spacing,IC)

    #Set final conditions
    IC = 0
    BC_f_h = boundary_conditions_human(human[-1,:,:], human_C[-1,:,:],human_LA[-1,:,:],human_RA[-1,:,:],human_LL[-1,:,:],human_RL[-1,:,:],Trajectory,grid_spacing,IC)
    
    #Add all components of human sweep together
    human = np.zeros((6,len(patches_hum[0])+len(patches_shift_C[0])+len(patches_shift_LA[0])+len(patches_shift_RA[0])+len(patches_shift_LL[0])+len(patches_shift_RL[0])+len(BC_i_h[0])+len(BC_f_h[0])))
    for ii in range(6):
        human[ii,:] = np.concatenate((patches_hum[ii],patches_shift_C[ii],patches_shift_LA[ii],patches_shift_RA[ii],patches_shift_LL[ii],patches_shift_RL[ii],BC_i_h[ii],BC_f_h[ii]), axis=None)
    
    #Determine the area swept out by robot. Data stored in XYZ coordinates within patches
    patches = generateSurface(points,Traj_rob,grid_spacing)
    patches_shift = generateSurface(points_shift,Traj_rob,grid_spacing)
    
    #Set Boundary conditions for robot
    #Set initial conditions
    IC = 1
    BC_i_r = boundary_conditions_robot(points[0,:,:], points_shift[0,:,:],Traj_rob,grid_spacing,IC)

    #Set final conditions
    IC = 0
    BC_f_r = boundary_conditions_robot(points[-1,:,:], points_shift[-1,:,:],Traj_rob,grid_spacing,IC)

    #Add all components of robot sweep together
    robot = np.zeros((6,len(patches[0])+len(patches_shift[0])+len(BC_i_r[0])+len(BC_f_r[0])))
    for ii in range(6):
        robot[ii,:] = np.concatenate((patches[ii],patches_shift[ii],BC_i_r[ii],BC_f_r[ii]), axis=None)
    
    #To be implemented when applied to live and predicted situation
    #np.save('robotSweep', robot)
    #robot = np.load('robotSweep.npy')
    #Compare two sweeps for collision
    
    collision, segments, collisionPoints = collisionCheck(robot,human,grid_spacing)
    
    if collision:
       print("Collision at segment(s) ", segments)

    return robot, human, collisionPoints

def swapYZ(boundary_points):
    Y = boundary_points[:,2].copy()
    Z = -boundary_points[:,1].copy()
    boundary_points[:,2] = Z
    boundary_points[:,1] = Y
    return boundary_points


if __name__ == "__main__":

    pipeline, align, unaligned_frames, frames, depth, depth_intrinsic, window_name, skeletrack, joint_confidence = realSenseSetup()
    humandata = defaultdict(list)
    ii = 0
    while True:
        start_time = time.time()
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        unaligned_frames = pipeline.wait_for_frames()
        frames = align.process(unaligned_frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth.get_data())
        color_image = np.asanyarray(color.get_data())

        # perform inference and update the tracking id
        skeletons = skeletrack.track_skeletons(color_image)
        if len(skeletons) > 0:
            # render the skeletons on top of the acquired image and display it
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            cmr.render_result(skeletons, color_image, joint_confidence)
            joint_location = render_ids_3d(color_image, skeletons, depth, depth_intrinsic, joint_confidence)
            humandata[ii] = joint_location
            ii+=1
            cv2.imshow(window_name, color_image)

            if cv2.waitKey(1) == 27:
                break
        if ii > 150:
            break

        print(time.time()-start_time)

    pipeline.stop()
    cv2.destroyAllWindows()
   
    with open('HumanMotionGood.pickle', 'wb') as handle:
        pickle.dump(humandata, handle, protocol=pickle.HIGHEST_PROTOCOL)
