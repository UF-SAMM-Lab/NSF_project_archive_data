#!/usr/bin/env python3
import numpy as np
import math
from collections import defaultdict
import pickle
#Sample Manipulator Trajectory 
def rob_traj(a,b,timea,timeb):
    
    #Number of desired steps
    steps = 5

    #Range of motion for each joint
    J2 = ([a[0],  b[0]])
    J3 = ([a[1],  b[1]])
    J4 = ([a[2],  b[2]])
    J5 = ([a[3],  b[3]])  
    J6 = ([a[4],  b[4]])
    
    #Joint position 2-7 in the array correspond to joint angles 1-6
    #Joint position 1 in the array corresponds to the base (no angle change)
    jointTrajectory = np.zeros(shape=(7,steps))
    jointTrajectory[1,:] = np.linspace(J2[0], J2[1], steps)
    jointTrajectory[2,:] = np.linspace(J3[0], J3[1], steps)
    jointTrajectory[3,:] = np.linspace(J4[0], J4[1], steps)   
    jointTrajectory[4,:] = np.linspace(J5[0], J5[1], steps)   
    jointTrajectory[5,:] = np.linspace(J6[0], J6[1], steps)
    jointTrajectory[6,:] = np.zeros((1,steps))

    #Hard coded time stamps and segment numbers
    Trajectory = np.zeros(shape=(2,steps))
    Trajectory[0:,] = np.linspace(timea, timeb, steps)
    Trajectory[1:,] = range(1,1+steps)

    return jointTrajectory, Trajectory

if __name__ == '__main__':
    schedule = defaultdict(list)
    
    a = np.array([0,0,0,0,0])
    b = np.array([math.pi/4,  math.pi/4,  math.pi/4,  0,0])
    c = np.array([math.pi/2,    math.pi/4,           0,  math.pi/4, math.pi/4])
    d = np.array([0,0,0,0,0])
    timea = 0
    timeb = 2
    timec = 4
    timed = 6
    jointTrajectory, Trajectory = rob_traj(a,b,timea,timeb)
    schedule[1].append(jointTrajectory)
    schedule[1].append(Trajectory)
    jointTrajectory, Trajectory = rob_traj(b,c,timeb,timec)
    schedule[2].append(jointTrajectory)
    schedule[2].append(Trajectory)
    jointTrajectory, Trajectory = rob_traj(c,d,timec,timed)
    schedule[3].append(jointTrajectory)
    schedule[3].append(Trajectory)
    '''
    a = np.array([ 0,0,0,0,0])
    b = np.array([math.pi,0,0,0,0])
    timea = 0
    timeb = 5
    jointTrajectory, Trajectory = rob_traj(a,b,timea,timeb)
    schedule[1].append(jointTrajectory)
    schedule[1].append(Trajectory)
    b = 0
    '''
    '''
    for ii in range(int((len(schedule)/2))):
        print(schedule[(2*ii+2)][0,0][0])
        print(schedule[(2*ii+2)][0,0][-1])
        #print(time[-1])
    ''' 
    
    with open('robotSchedule_sweep.pickle', 'wb') as handle:
        pickle.dump(schedule, handle, protocol=pickle.HIGHEST_PROTOCOL)



    