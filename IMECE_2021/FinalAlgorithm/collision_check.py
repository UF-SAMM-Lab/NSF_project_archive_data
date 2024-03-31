#!/usr/bin/env python
import numpy as np
import pandas as pd
from numba import jit

@jit(nopython=True)
def mapToGrid(grid_spacing,pc):
    #The purpose of this fuction is to determine the closest node to each point
     #in the given array.
    for ii in range(3):
        for jj in range(len(pc[0])):
            if pc[ii,jj] > 0:
                rem = np.mod(pc[ii,jj],grid_spacing)
                if rem > grid_spacing/2:
                    pc[ii,jj] = round(pc[ii,jj] + (grid_spacing - rem),2)
                else:
                    pc[ii,jj] = round(pc[ii,jj] - rem,2)
            else:
                rem = np.mod(abs(pc[ii,jj]),grid_spacing)
                if rem > grid_spacing/2:
                    pc[ii,jj] = round(pc[ii,jj] - (grid_spacing - rem),2)
                else:
                    pc[ii,jj] = round(pc[ii,jj] + rem,2)
    return pc

@jit(nopython=True)
def mapToTimeGrid(time_spacing,time):
    #The purpose of this fuction is to determine the closest node to each point
     #in the given array.
    for jj in range(len(time)):
        rem = np.mod(time[jj],time_spacing)
        if rem > time_spacing/2:
            time[jj] =round(time[jj] + (time_spacing - rem),2)
        else:
            time[jj] = time[jj] - rem

    return time

@jit(nopython=True)
def Descritize_XYZ(grid_spacing,pc):
    #The purpose of this fuction is to determine the closest node to each point
     #in the given array.
        
    #---------------------------------------------------------
    #Calculate the lower bound gridpoint for each direction
    #---------------------------------------------------------
    Xl = pc[0] - np.mod(pc[0],grid_spacing)
    Yl = pc[1] - np.mod(pc[1],grid_spacing)
    Zl = pc[2] - np.mod(pc[2],grid_spacing)
    
    #---------------------------------------------------------
    #Calculate the upper bound gridpoint for each direction
    #---------------------------------------------------------
    Xu = Xl + grid_spacing
    Yu = Xl + grid_spacing
    Zu = Xl + grid_spacing
    
    #---------------------------------------------------------
    #Select the node that is closest
    #---------------------------------------------------------      
    #comapare is a binary indicator. If the real point is closer to the
    #upper point, a 1 will be returned
    discrete = np.zeros((3,len(pc[0])))
    compareX = Xu - pc[0] < pc[0] - Xl
    discrete[0] = Xl + compareX*grid_spacing
    
    compareY = Yu - pc[1] < pc[1] - Yl
    discrete[1] = Yl + compareY*grid_spacing
    
    compareZ = Zu - pc[2] < pc[2] - Zl
    discrete[2] = Zl + compareZ*grid_spacing 

    return discrete

@jit(nopython=True)
def Descritize_time(time_step, anylitical):
    #The purpose of this fuction is to determine the closest standard point
    #in time in the given array.
        
        #---------------------------------------------------------
        #Calculate the lower bound time
        #---------------------------------------------------------
        Tl = anylitical - np.mod(anylitical,time_step)
        
        #---------------------------------------------------------
        #Calculate the upper bound time
        #---------------------------------------------------------
        Tu = Tl + time_step
        
        #---------------------------------------------------------
        #Select the time that is closest
        #---------------------------------------------------------      
        #comapare is a binary indicator. If the real time is closer to the
        #upper bound time, a 1 will be returned
        compareT = (Tl - anylitical) < (anylitical - Tu)
        discrete = Tl + compareT*time_step

        return discrete   

def collisionCheck(data1, data2, grid_spacing):
    #--------------------------------------------------------------------------
    #The purpose of this function is to compare the data between the surface
    #generated from the robot trajectory and the cone of uncertainty from the
    #obstacle

    #The comparison is accomplished by descritizing the points in each data
    #set, inputting them into cartesian grids, and comparing the grids
    #--------------------------------------------------------------------------

    #Convert data to a grid with a known spacing so that they are comparable
    grid1 = Descritize_XYZ(grid_spacing, data1)  
    grid2 = Descritize_XYZ(grid_spacing, data2)

    #Descritize data time stamp into compare them more accurately
    time_step = 0.25
    time1 = Descritize_time(time_step, data1[3])
    time2 = Descritize_time(time_step, data2[3])

    #Recast into a form that would allow for quick lookup of collision info
    cartesian_set1 = np.zeros((4,len(grid1[0])))
    cartesian_set2 = np.zeros((4,len(grid2[0])))
    for ii in range(3):
        cartesian_set1[ii,:] = grid1[ii]
        cartesian_set2[ii,:] = grid2[ii]

    cartesian_set1[3,:] = time1
    cartesian_set2[3,:] = time2
    d1 = {'x': grid1[0], 'y': grid1[1], 'z': grid1[2], 't': time1, 'seg_1': data1[4]}
    df1 = pd.DataFrame(data=d1)
    d2 = {'x': grid2[0], 'y': grid2[1], 'z': grid2[2], 't': time2, 'seg_2': data2[4]}
    df2 = pd.DataFrame(data=d2)

    #collision points
    overlay = pd.merge(df1,df2,on=['x','y','z','t'], how='inner')

    #Convert dataframe to numpy array
    overlay = overlay.to_numpy()
    if len(overlay) > 0:
        segments = np.unique(overlay[:,4])
        collision = True
    else:
        segments = []
        collision = False

    return collision, segments, overlay

