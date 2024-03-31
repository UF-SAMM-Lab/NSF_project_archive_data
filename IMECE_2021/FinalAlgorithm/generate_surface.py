#!/usr/bin/env python
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import jit

@jit(nopython=True)
def mesh_setter(points, ii, jj, grid_spacing):
    #The purpose of this function is to evaluate the size of the patch to be
    #created by the four points and set the mesh so that its spacing is at most half the
    #size of the overall grid spacing.
#------------------------------- CURRENT PROGRESS ------------------------------------
###CHECK INDEXING
    #Evaluate the XYZ Distance between each point
    line1 = math.sqrt( (points[ii+1,0,jj  ] - points[ii,  0,jj  ])**2 + (points[ii+1,1,jj  ] - points[ii,  1,jj  ])**2 + (points[ii+1,2,jj  ] - points[ii,  2,jj  ])**2)
    line2 = math.sqrt( (points[ii+1,0,jj-1] - points[ii+1,0,jj  ])**2 + (points[ii+1,1,jj-1] - points[ii+1,1,jj  ])**2 + (points[ii+1,2,jj-1] - points[ii+1,2,jj  ])**2)
    line3 = math.sqrt( (points[ii+1,0,jj-1] - points[ii,  0,jj-1])**2 + (points[ii+1,1,jj-1] - points[ii,  1,jj-1])**2 + (points[ii+1,2,jj-1] - points[ii,  2,jj-1])**2)
    line4 = math.sqrt( (points[ii,  0,jj  ] - points[ii,  0,jj-1])**2 + (points[ii,  1,jj  ] - points[ii,  1,jj-1])**2 + (points[ii,  2,jj  ] - points[ii,  2,jj-1])**2)

    #Use the longest boundary curve to define the mesh
    longest_u = max(line1,line3)
    nbf_u = math.ceil(longest_u/(grid_spacing))
    longest_v = max(line2,line4)
    nbf_v = math.ceil(longest_v/(grid_spacing))

    #In order to establish a normal, the mesh must be greated than 1 x 1
    if nbf_u <= 1:
        nbf_u = 2

    if nbf_v <= 1:
        nbf_v = 2

    return nbf_u, nbf_v

def generateSurface(points, Trajectory, grid_spacing):

    #Set point size for coons patches
    patches_x = np.array([])
    patches_y = np.array([])
    patches_z = np.array([])
    patches_time = np.array([])
    patches_segment = np.array([])
    patches_joint = np.array([])

    #This function calls for patches to be made then organizes them into a data structure
    patches_x, patches_y, patches_z, patches_time, patches_segment, patches_joint = generateSurfaceMath(points, Trajectory, grid_spacing, patches_x, patches_y, patches_z, patches_time, patches_segment, patches_joint)

    #Recast data into a stucture
    patches = np.zeros((6,len(patches_x)))
    patches[0,:] = patches_x
    patches[1,:] = patches_y
    patches[2,:] = patches_z
    patches[3,:] = patches_time
    patches[4,:] = patches_segment
    patches[5,:] = patches_joint

    return patches

@jit(nopython=True)
def generateSurfaceMath(points, Trajectory, grid_spacing, patches_x, patches_y, patches_z, patches_time, patches_segment, patches_joint):
    #--------------------------------------------------------------------------
    #The purpose of this function is create coon's patch using a set of points.
    #--------------------------------------------------------------------------
    
    #Reshape into curves to feed into coons patches algorithm
    a = len(points)
    b = len(points[0][0])

    for ii in range(0,a-1):  #Step through each region between poses
        for jj in range(b-1,0,-1): #Step through each region between joints
            #Determine required mesh size according to size of the patch
            [nbf_u,nbf_v] = mesh_setter(points, ii, jj, grid_spacing)

            #Define boundary curves for Coons Patch
            #Initialize curves
            c1 = np.zeros((3,nbf_u,1))
            c2 = np.zeros((3,nbf_v,1))
            c3 = np.zeros((3,nbf_u,1))
            c4 = np.zeros((3,nbf_v,1))
            
            #Curve 1
            c1[0,0,0] = points[ii,0,jj]
            c1[1,0,0] = points[ii,1,jj]
            c1[2,0,0] = points[ii,2,jj]
            c1[0,nbf_u-1,0] = points[ii+1,0,jj]
            c1[1,nbf_u-1,0] = points[ii+1,1,jj]
            c1[2,nbf_u-1,0] = points[ii+1,2,jj]

            x1 = c1[0,0,0]
            y1 = c1[1,0,0]
            z1 = c1[2,0,0]
            x2 = c1[0,-1,0]
            y2 = c1[1,-1,0]
            z2 = c1[2,-1,0]

            xspace = (x2 - x1)/(nbf_u-1)
            yspace = (y2 - y1)/(nbf_u-1)
            zspace = (z2 - z1)/(nbf_u-1)

            for aa in range(1,nbf_u):
                c1[0,aa,0] = c1[0,0,0] + xspace*(aa)
                c1[1,aa,0] = c1[1,0,0] + yspace*(aa)
                c1[2,aa,0] = c1[2,0,0] + zspace*(aa)

            #Curve 3
            c3[0,0,0] = points[ii,0,jj-1]
            c3[1,0,0] = points[ii,1,jj-1]
            c3[2,0,0] = points[ii,2,jj-1]
            c3[0,nbf_u-1,0] = points[ii+1,0,jj-1]
            c3[1,nbf_u-1,0] = points[ii+1,1,jj-1]
            c3[2,nbf_u-1,0] = points[ii+1,2,jj-1]
            
            x1 = c3[0,0,0]
            y1 = c3[1,0,0]
            z1 = c3[2,0,0]
            x2 = c3[0,-1,0]
            y2 = c3[1,-1,0]
            z2 = c3[2,-1,0]

            xspace = (x2 - x1)/(nbf_u-1)
            yspace = (y2 - y1)/(nbf_u-1)
            zspace = (z2 - z1)/(nbf_u-1)

            for aa in range(1,nbf_u):
                c3[0,aa,0] = c3[0,0,0] + xspace*(aa)
                c3[1,aa,0] = c3[1,0,0] + yspace*(aa)
                c3[2,aa,0] = c3[2,0,0] + zspace*(aa)

            #Curve 2
            c2[0,0,0] = c1[0,nbf_u-1,0]
            c2[1,0,0] = c1[1,nbf_u-1,0]
            c2[2,0,0] = c1[2,nbf_u-1,0]
            c2[0,nbf_v-1,0] = c3[0,nbf_u-1,0]
            c2[1,nbf_v-1,0] = c3[1,nbf_u-1,0]
            c2[2,nbf_v-1,0] = c3[2,nbf_u-1,0]
            
            x1 = c2[0,0,0]
            y1 = c2[1,0,0]
            z1 = c2[2,0,0]
            x2 = c2[0,-1,0]
            y2 = c2[1,-1,0]
            z2 = c2[2,-1,0]

            xspace = (x2 - x1)/(nbf_v-1)
            yspace = (y2 - y1)/(nbf_v-1)
            zspace = (z2 - z1)/(nbf_v-1)

            for aa in range(1,nbf_v):
                c2[0,aa,0] = c2[0,0,0] + xspace*(aa)
                c2[1,aa,0] = c2[1,0,0] + yspace*(aa)
                c2[2,aa,0] = c2[2,0,0] + zspace*(aa)

            #Curve 4
            c4[0,0,0] = c1[0,0,0]
            c4[1,0,0] = c1[1,0,0]
            c4[2,0,0] = c1[2,0,0]
            c4[0,nbf_v-1,0] = c3[0,0,0]
            c4[1,nbf_v-1,0] = c3[1,0,0]
            c4[2,nbf_v-1,0] = c3[2,0,0]
            
            x1 = c4[0,0,0]
            y1 = c4[1,0,0]
            z1 = c4[2,0,0]
            x2 = c4[0,-1,0]
            y2 = c4[1,-1,0]
            z2 = c4[2,-1,0]

            xspace = (x2 - x1)/(nbf_v-1)
            yspace = (y2 - y1)/(nbf_v-1)
            zspace = (z2 - z1)/(nbf_v-1)

            for aa in range(1,nbf_v):
                c4[0,aa,0] = c4[0,0,0] + xspace*(aa)
                c4[1,aa,0] = c4[1,0,0] + yspace*(aa)
                c4[2,aa,0] = c4[2,0,0] + zspace*(aa)

            #Create patches to describe the position thorughout the surface
            #Set up indices for each spot in Coons patch
            c1_spl_vect = range(1,nbf_u+1)
            c3_spl_vect = range(1,nbf_u+1)
            c2_spl_vect = range(1,nbf_v+1)
            c4_spl_vect = range(1,nbf_v+1)
            
            #Translate curves to Coons Patch
            Coons_patch = np.zeros(shape=(3,nbf_u,nbf_v))
            for aa in c1_spl_vect:
                Coons_patch[:,nbf_u-aa,0]  = c1[:,aa-1,0]   
            for aa in c3_spl_vect:
                Coons_patch[:,nbf_u-aa,-1] = c3[:,aa-1,0]
            for aa in c2_spl_vect:
                Coons_patch[:,0,aa-1]  = c2[:,aa-1,0]
            for aa in c4_spl_vect:
                Coons_patch[:,-1,aa-1] = c4[:,aa-1,0]
            
            Coons_patch[:,0,0]  = c2[:,0,0]
            Coons_patch[:,-1,0]  = c4[:,0,0]

            #Patch values computation
            for n in range(1,nbf_u+1):
                for p in range(1,nbf_v+1):
                    
                    u = (n-1)/(nbf_u-1)
                    v = (p-1)/(nbf_v-1)
            
                    Coons_patch[0,n-1,p-1] = (1-u)*Coons_patch[0,0,p-1]+(1-v)*Coons_patch[0,n-1,0]+u*Coons_patch[0,-1,p-1]+v*Coons_patch[0,n-1,-1]+(u-1)*(1-v)*Coons_patch[0,0,0]-u*v*Coons_patch[0,-1,-1]+u*(v-1)*Coons_patch[0,-1,0]+v*(u-1)*Coons_patch[0,0,-1]
                    Coons_patch[1,n-1,p-1] = (1-u)*Coons_patch[1,0,p-1]+(1-v)*Coons_patch[1,n-1,0]+u*Coons_patch[1,-1,p-1]+v*Coons_patch[1,n-1,-1]+(u-1)*(1-v)*Coons_patch[1,0,0]-u*v*Coons_patch[1,-1,-1]+u*(v-1)*Coons_patch[1,-1,0]+v*(u-1)*Coons_patch[1,0,-1]
                    Coons_patch[2,n-1,p-1] = (1-u)*Coons_patch[2,0,p-1]+(1-v)*Coons_patch[2,n-1,0]+u*Coons_patch[2,-1,p-1]+v*Coons_patch[2,n-1,-1]+(u-1)*(1-v)*Coons_patch[2,0,0]-u*v*Coons_patch[2,-1,-1]+u*(v-1)*Coons_patch[2,-1,0]+v*(u-1)*Coons_patch[2,0,-1]
            


            #Create a patch to describe the time throughout the surface
            time = np.zeros((1,nbf_u));            
            time[0,0] = Trajectory[0,ii+1]
            time[0,-1] = Trajectory[0,ii]
            t1 = time[0,0]
            t2 = time[0,-1]

            tspace = (t2 - t1)/(nbf_u-1)
            for aa in range(1,nbf_u-1):
                time[0,aa] = time[0,0] + tspace*(aa)

            T = np.zeros(shape=(len(time[0]),nbf_v))
            for p in range(0,nbf_v):
                T[:,p] = time

            #Deconstruct Patch into arrays
            for kk in range(len(Coons_patch[0])):
                patches_x       = np.concatenate((patches_x,       Coons_patch[0,kk,:]))
                patches_y       = np.concatenate((patches_y,       Coons_patch[1,kk,:]))   
                patches_z       = np.concatenate((patches_z,       Coons_patch[2,kk,:]))
                patches_time    = np.concatenate((patches_time,    T[kk,:]))
                patches_segment = np.concatenate((patches_segment, np.ones(len(Coons_patch[0,kk,:]))*Trajectory[1,ii]))
                patches_joint   = np.concatenate((patches_joint,   np.ones(len(Coons_patch[0,kk,:]))*jj))
                
    return patches_x, patches_y, patches_z, patches_time, patches_segment, patches_joint

def boundary_conditions_robot(C,B,Trajectory_in,grid_spacing,IC):
    #The purpose of this function is to set an initial and final condition
    #closed volume on the robot
    
    #Use Generate Surface funciton to create patches around the first link
    points_a = np.zeros((5,4,2))
    points_a[0,:,0] = C[:,0]   
    points_a[0,:,1] = C[:,1] 
    points_a[1,:,0] = B[:,0]   
    points_a[1,:,1] = B[:,1] 
    points_a[2,:,0] = C[:,11]  
    points_a[2,:,1] = C[:,10] 
    points_a[3,:,0] = B[:,11]  
    points_a[3,:,1] = B[:,10] 
    points_a[4,:,0] = C[:,0]   
    points_a[4,:,1] = C[:,1]

    Trajectory = np.zeros((2,5)) 
    Trajectory[1,:] = 1
    if IC == 1:
        Trajectory[0,:] = Trajectory_in[0,0]
        Trajectory[1,:] = Trajectory_in[1,0]
    else:
        Trajectory[0,:] = Trajectory_in[0,-1]
        #Subtract 1 because the last value correspends to the ending waypoint of the segment
        #not the segment itself
        Trajectory[1,:] = Trajectory_in[1,-1] - 1
    

    link_one = generateSurface(points_a,Trajectory,grid_spacing)
    #Use Generate Surface funciton to create patches around link 2
    points_b = np.zeros((5,4,2))
    points_b[0,:,0] = C[:,1]   
    points_b[0,:,1] = C[:,2] 
    points_b[1,:,0] = B[:,1]   
    points_b[1,:,1] = B[:,2] 
    points_b[2,:,0] = C[:,10]  
    points_b[2,:,1] = C[:,9] 
    points_b[3,:,0] = B[:,10] 
    points_b[3,:,1] = B[:,9] 
    points_b[4,:,0] = C[:,1]   
    points_b[4,:,1] = C[:,1]    

    link_two = generateSurface(points_b,Trajectory,grid_spacing)
    #Use Generate Surface funciton to create patches around link 3
    points_c = np.zeros((5,4,2))
    points_c[0,:,0] = C[:,2]   
    points_c[0,:,1] = C[:,3] 
    points_c[1,:,0] = B[:,2]   
    points_c[1,:,1] = B[:,3] 
    points_c[2,:,0] = C[:,9]   
    points_c[2,:,1] = C[:,8] 
    points_c[3,:,0] = B[:,9]   
    points_c[3,:,1] = B[:,8] 
    points_c[4,:,0] = C[:,2]   
    points_c[4,:,1] = C[:,3] 

    link_three = generateSurface(points_c,Trajectory,grid_spacing)
    #Use Generate Surface funciton to create patches around link 4
    points_d = np.zeros((5,4,2))
    points_d[0,:,0] = C[:,3]   
    points_d[0,:,1] = C[:,4] 
    points_d[1,:,0] = B[:,3]   
    points_d[1,:,1] = B[:,4] 
    points_d[2,:,0] = C[:,8]   
    points_d[2,:,1] = C[:,7] 
    points_d[3,:,0] = B[:,8]   
    points_d[3,:,1] = B[:,7] 
    points_d[4,:,0] = C[:,3]   
    points_d[4,:,1] = C[:,4] 
    
    link_four = generateSurface(points_d,Trajectory,grid_spacing)
    #Use Generate Surface funciton to create patches around link 5
    points_e = np.zeros((5,4,2))
    points_e[0,:,0] = C[:,4]   
    points_e[0,:,1] = C[:,5] 
    points_e[1,:,0] = B[:,4]   
    points_e[1,:,1] = B[:,5] 
    points_e[2,:,0] = C[:,7]   
    points_e[2,:,1] = C[:,6] 
    points_e[3,:,0] = B[:,7]   
    points_e[3,:,1] = B[:,6] 
    points_e[4,:,0] = C[:,4]   
    points_e[4,:,1] = C[:,5] 

    link_five = generateSurface(points_e,Trajectory,grid_spacing)
    #Package data for output
    robot = np.zeros((6,len(link_one[0]) + len(link_two[0]) + len(link_three[0]) + len(link_four[0]) + len(link_five[0])))
    robot[0,:] = np.concatenate((link_one[0], link_two[0], link_three[0], link_four[0], link_five[0]), axis=None)
    robot[1,:] = np.concatenate((link_one[1], link_two[1], link_three[1], link_four[1], link_five[1]), axis=None)
    robot[2,:] = np.concatenate((link_one[2], link_two[2], link_three[2], link_four[2], link_five[2]), axis=None)
    robot[3,:] = np.concatenate((link_one[3], link_two[3], link_three[3], link_four[3], link_five[3]), axis=None)
    robot[4,:] = np.concatenate((link_one[4], link_two[4], link_three[4], link_four[4], link_five[4]), axis=None)
    robot[5,:] = np.concatenate((link_one[5], link_two[5], link_three[5], link_four[5], link_five[5]), axis=None)
    
    return robot

@jit(nopython=True)
def homCoords(points):
    a, b = np.shape(points)
    homPoints = np.zeros((a+1,b))
    homPoints[0,:] = points[0,:]
    homPoints[1,:] = points[1,:]
    homPoints[2,:] = points[2,:]
    homPoints[3,:] = np.ones(np.shape(points[0,:]))
    return homPoints


def boundary_conditions_human(B,C,RA,LA,RL,LL,Trajectory_in,grid_spacing,IC,legs):
    #The purpose of this function is to set an initial and final condition
    #closed volume on the human

    #Convert to homogeneous coordinates
    B = homCoords(B)
    C = homCoords(C)
    LA = homCoords(LA)
    RA = homCoords(RA)
    if legs:
        LL = homCoords(LL)
        RL = homCoords(RL)

    #Use Generate Surface funciton to create patches around the head
    points_H = np.zeros((5,4,2))
    points_H[0,:,0] = C[:,2]   
    points_H[0,:,1] = C[:,1] 
    points_H[1,:,0] = B[:,9]   
    points_H[1,:,1] = B[:,10] 
    points_H[2,:,0] = C[:,3]  
    points_H[2,:,1] = C[:,4] 
    points_H[3,:,0] = B[:,8]  
    points_H[3,:,1] = B[:,7] 
    points_H[4,:,0] = C[:,2]   
    points_H[4,:,1] = C[:,1]

    Trajectory = np.zeros((2,len(points_H))) 
    Trajectory[1,:] = 1
    if IC == 1:
        Trajectory[0,:] = Trajectory_in[0,0]
        Trajectory[1,:] = Trajectory_in[1,0]
    else:
        Trajectory[0,:] = Trajectory_in[0,-1]
        #Subtract 1 because the last value correspends to the ending waypoint of the segment
        #not the segment itself
        Trajectory[1,:] = Trajectory_in[1,-1] - 1
    Head = generateSurface(points_H,Trajectory,grid_spacing)

    #Use Generate Surface funciton to create patches around left arm
    points_LA = np.zeros((5,4,3))
    points_LA[0,:,0] = LA[:,0]   
    points_LA[0,:,1] = LA[:,1] 
    points_LA[0,:,2] = LA[:,2] 
    points_LA[1,:,0] = B[:,11]   
    points_LA[1,:,1] = B[:,12]
    points_LA[1,:,2] = B[:,13]
    points_LA[2,:,0] = LA[:,5]   
    points_LA[2,:,1] = LA[:,4]
    points_LA[2,:,2] = LA[:,3]
    points_LA[3,:,0] = B[:,16]   
    points_LA[3,:,1] = B[:,15]
    points_LA[3,:,2] = B[:,14]
    points_LA[4,:,0] = LA[:,0]   
    points_LA[4,:,1] = LA[:,1] 
    points_LA[4,:,2] = LA[:,2]

    LeftArm = generateSurface(points_LA,Trajectory,grid_spacing)

    #Use Generate Surface funciton to create patches around right arm
    points_RA = np.zeros((5,4,3))
    points_RA[0,:,0] = RA[:,0]   
    points_RA[0,:,1] = RA[:,1] 
    points_RA[0,:,2] = RA[:,2] 
    points_RA[1,:,0] = B[:,1]   
    points_RA[1,:,1] = B[:,2]
    points_RA[1,:,2] = B[:,3]
    points_RA[2,:,0] = RA[:,5]   
    points_RA[2,:,1] = RA[:,4]
    points_RA[2,:,2] = RA[:,3]
    points_RA[3,:,0] = B[:,6]   
    points_RA[3,:,1] = B[:,5]
    points_RA[3,:,2] = B[:,4]
    points_RA[4,:,0] = RA[:,0]   
    points_RA[4,:,1] = RA[:,1] 
    points_RA[4,:,2] = RA[:,2]

    RightArm = generateSurface(points_RA,Trajectory,grid_spacing)
    if legs:
        #Use Generate Surface funciton to create patches around left leg
        points_LL = np.zeros((5,4,3))
        points_LL[0,:,0] = LL[:,0]   
        points_LL[0,:,1] = LL[:,1] 
        points_LL[0,:,2] = LL[:,2] 
        points_LL[1,:,0] = B[:,17]   
        points_LL[1,:,1] = B[:,18]
        points_LL[1,:,2] = B[:,19]
        points_LL[2,:,0] = LL[:,5]   
        points_LL[2,:,1] = LL[:,4]
        points_LL[2,:,2] = LL[:,3]
        points_LL[3,:,0] = B[:,22]   
        points_LL[3,:,1] = B[:,21]
        points_LL[3,:,2] = B[:,20]
        points_LL[4,:,0] = LL[:,0]   
        points_LL[4,:,1] = LL[:,1] 
        points_LL[4,:,2] = LL[:,2]

        LeftLeg = generateSurface(points_LL,Trajectory,grid_spacing)
        
        #Use Generate Surface funciton to create patches around right leg
        points_RL = np.zeros((5,4,3))
        points_RL[0,:,0] = RL[:,0]   
        points_RL[0,:,1] = RL[:,1] 
        points_RL[0,:,2] = RL[:,2] 
        points_RL[1,:,0] = B[:,22]   
        points_RL[1,:,1] = B[:,23]
        points_RL[1,:,2] = B[:,24]
        points_RL[2,:,0] = RL[:,5]   
        points_RL[2,:,1] = RL[:,4]
        points_RL[2,:,2] = RL[:,3]
        points_RL[3,:,0] = B[:,0]   
        points_RL[3,:,1] = B[:,26]
        points_RL[3,:,2] = B[:,25]
        points_RL[4,:,0] = RL[:,0]   
        points_RL[4,:,1] = RL[:,1] 
        points_RL[4,:,2] = RL[:,2]

        RightLeg = generateSurface(points_RL,Trajectory,grid_spacing)

    #Use Generate Surface funciton to create patches around right half of body
    points_CR = np.zeros((7,4,2))
    points_CR[0,:,0] = B[:,0]   
    points_CR[0,:,1] = C[:,0] 
    points_CR[1,:,0] = B[:,1]   
    points_CR[1,:,1] = C[:,1]
    points_CR[2,:,0] = RA[:,0]   
    points_CR[2,:,1] = B[:,7]
    points_CR[3,:,0] = B[:,6]   
    points_CR[3,:,1] = B[:,7]
    points_CR[4,:,0] = RA[:,5]   
    points_CR[4,:,1] = B[:,7] 
    points_CR[5,:,0] = B[:,1]   
    points_CR[5,:,1] = C[:,4]
    points_CR[6,:,0] = B[:,0]   
    points_CR[6,:,1] = C[:,5] 

    Trajectory = np.zeros((2,len(points_CR))) 
    Trajectory[1,:] = 1
    if IC == 1:
        Trajectory[0,:] = Trajectory_in[0,0]
        Trajectory[1,:] = Trajectory_in[1,0]
    else:
        Trajectory[0,:] = Trajectory_in[0,-1]
        #Subtract 1 because the last value correspends to the ending waypoint of the segment
        #not the segment itself
        Trajectory[1,:] = Trajectory_in[1,-1] - 1
 
    CenterRight = generateSurface(points_CR,Trajectory,grid_spacing)

    #Use Generate Surface funciton to create patches around left half of body
    points_CL = np.zeros((7,4,2))
    points_CL[0,:,0] = B[:,17]   
    points_CL[0,:,1] = C[:,0] 
    points_CL[1,:,0] = B[:,16]   
    points_CL[1,:,1] = C[:,1]
    points_CL[2,:,0] = LA[:,0]   
    points_CL[2,:,1] = B[:,10]
    points_CL[3,:,0] = B[:,11]   
    points_CL[3,:,1] = B[:,10]
    points_CL[4,:,0] = LA[:,5]   
    points_CL[4,:,1] = B[:,10] 
    points_CL[5,:,0] = B[:,16]   
    points_CL[5,:,1] = C[:,4]
    points_CL[6,:,0] = B[:,17]   
    points_CL[6,:,1] = C[:,5]

    CenterLeft = generateSurface(points_CL,Trajectory,grid_spacing)

    #Generate surfaces to cap off the hands, feet, and top of head
    points_HL = np.zeros((2,4,2))
    points_HL[0,:,0] = B[:,14]   
    points_HL[0,:,1] = LA[:,2]
    points_HL[1,:,0] = LA[:,3]   
    points_HL[1,:,1] = B[:,13]

    Trajectory = np.zeros((2,len(points_HL))) 
    Trajectory[1,:] = 1
    if IC == 1:
        Trajectory[0,:] = Trajectory_in[0,0]
        Trajectory[1,:] = Trajectory_in[1,0]
    else:
        Trajectory[0,:] = Trajectory_in[0,-1]
        #Subtract 1 because the last value correspends to the ending waypoint of the segment
        #not the segment itself
        Trajectory[1,:] = Trajectory_in[1,-1] - 1

    HandLeft = generateSurface(points_HL,Trajectory,grid_spacing)

    points_HR = np.zeros((2,4,2))
    points_HR[0,:,0] = B[:,4]   
    points_HR[0,:,1] = RA[:,2]
    points_HR[1,:,0] = RA[:,3]   
    points_HR[1,:,1] = B[:,3]
    HandRight = generateSurface(points_HR,Trajectory,grid_spacing)
    
    if legs:
        points_FL = np.zeros((2,4,2))
        points_FL[0,:,0] = B[:,20]   
        points_FL[0,:,1] = LL[:,2]
        points_FL[1,:,0] = LL[:,3]   
        points_FL[1,:,1] = B[:,19]
        FootLeft = generateSurface(points_FL,Trajectory,grid_spacing)

        points_FR = np.zeros((2,4,2))
        points_FR[0,:,0] = B[:,25]   
        points_FR[0,:,1] = RL[:,2]
        points_FR[1,:,0] = RL[:,3]   
        points_FR[1,:,1] = B[:,24]
        FootRight = generateSurface(points_FR,Trajectory,grid_spacing)

    points_HT = np.zeros((2,4,2))
    points_HT[0,:,0] = B[:,8]   
    points_HT[0,:,1] = C[:,2]
    points_HT[1,:,0] = C[:,3]   
    points_HT[1,:,1] = B[:,9]
    headTop = generateSurface(points_HT,Trajectory,grid_spacing)

    #Package data for output
    if legs:
        human = np.zeros((6,len(Head[0]) + len(headTop[0])  + len(FootRight[0])  + len(FootLeft[0]) + len(HandRight[0])  + len(HandLeft[0]) + len(LeftArm[0]) + len(RightArm[0])  + len(LeftLeg[0]) + len(RightLeg[0]) + len(CenterRight[0]) + len(CenterLeft[0])))
        human[0,:] = np.concatenate((RightArm[0], LeftArm[0], RightLeg[0], LeftLeg[0], headTop[0], FootRight[0], FootLeft[0], HandRight[0], HandLeft[0], Head[0], CenterRight[0], CenterLeft[0]), axis=None)
        human[1,:] = np.concatenate((RightArm[1], LeftArm[1], RightLeg[1], LeftLeg[1], headTop[1], FootRight[1], FootLeft[1], HandRight[1], HandLeft[1], Head[1], CenterRight[1], CenterLeft[1]), axis=None)
        human[2,:] = np.concatenate((RightArm[2], LeftArm[2], RightLeg[2], LeftLeg[2], headTop[2], FootRight[2], FootLeft[2], HandRight[2], HandLeft[2], Head[2], CenterRight[2], CenterLeft[2]), axis=None)
        human[3,:] = np.concatenate((RightArm[3], LeftArm[3], RightLeg[3], LeftLeg[3], headTop[3], FootRight[3], FootLeft[3], HandRight[3], HandLeft[3], Head[3], CenterRight[3], CenterLeft[3]), axis=None)
        human[4,:] = np.concatenate((RightArm[4], LeftArm[4], RightLeg[4], LeftLeg[4], headTop[4], FootRight[4], FootLeft[4], HandRight[4], HandLeft[4], Head[4], CenterRight[4], CenterLeft[4]), axis=None)
        human[5,:] = np.concatenate((RightArm[5], LeftArm[5], RightLeg[5], LeftLeg[5], headTop[5], FootRight[5], FootLeft[5], HandRight[5], HandLeft[5], Head[5], CenterRight[5], CenterLeft[5]), axis=None)
    else:
        human = np.zeros((6,len(Head[0]) + len(headTop[0])  + len(HandRight[0])  + len(HandLeft[0]) + len(LeftArm[0]) + len(RightArm[0])  +  len(CenterRight[0]) + len(CenterLeft[0])))
        human[0,:] = np.concatenate((RightArm[0], LeftArm[0], headTop[0], HandRight[0], HandLeft[0], Head[0], CenterRight[0], CenterLeft[0]), axis=None)
        human[1,:] = np.concatenate((RightArm[1], LeftArm[1], headTop[1], HandRight[1], HandLeft[1], Head[1], CenterRight[1], CenterLeft[1]), axis=None)
        human[2,:] = np.concatenate((RightArm[2], LeftArm[2], headTop[2], HandRight[2], HandLeft[2], Head[2], CenterRight[2], CenterLeft[2]), axis=None)
        human[3,:] = np.concatenate((RightArm[3], LeftArm[3], headTop[3], HandRight[3], HandLeft[3], Head[3], CenterRight[3], CenterLeft[3]), axis=None)
        human[4,:] = np.concatenate((RightArm[4], LeftArm[4], headTop[4], HandRight[4], HandLeft[4], Head[4], CenterRight[4], CenterLeft[4]), axis=None)
        human[5,:] = np.concatenate((RightArm[5], LeftArm[5], headTop[5], HandRight[5], HandLeft[5], Head[5], CenterRight[5], CenterLeft[5]), axis=None)

    return human