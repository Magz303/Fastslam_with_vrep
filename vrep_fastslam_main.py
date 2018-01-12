#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Dec 28 11:35:05 2017

@author: magnus tarle
"""

# Start Vrep simulation and run this program.
print ('Robot applied estimation program started')
import threading # used to control the robot path in parallel to the observations
import numpy as np # used for numerical calculations
import matplotlib.pyplot as plt # used for plotting the robot position and map
import time # used for simulation time steps

# This file uses the vrep API library. Check that connection to Vrep can be established
try:
    import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

# Robot path drive in separate class which includes threading
import robotctrl

# vrep fastslam defined functions in separate module 
import vrep_fastslam_functions as vf

# Connect to remote API server (vrep)
vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5) # Connect to V-REP (remember to add this port to V-REP script)
if clientID!=-1:
    print ('Connected to remote API server')

# Get Vrep scene object handles for input and output of information
error_code, h_motor_left = vrep.simxGetObjectHandle(clientID, 'motor_front_left', vrep.simx_opmode_oneshot_wait)
error_code, h_motor_right = vrep.simxGetObjectHandle(clientID, 'motor_front_right', vrep.simx_opmode_oneshot_wait)
error_code, h_prox_sensor = vrep.simxGetObjectHandle(clientID, 'proximity_sensor', vrep.simx_opmode_oneshot_wait)
error_code, h_car_cam = vrep.simxGetObjectHandle(clientID, 'car_cam', vrep.simx_opmode_oneshot_wait)
error_code, h_car = vrep.simxGetObjectHandle(clientID, 'car', vrep.simx_opmode_oneshot_wait)
# CONSIDER: using for loop for handles and feature array
error_code, h_feature1 = vrep.simxGetObjectHandle(clientID, 'feature1', vrep.simx_opmode_oneshot_wait)
error_code, h_feature2 = vrep.simxGetObjectHandle(clientID, 'feature2', vrep.simx_opmode_oneshot_wait)
error_code, h_feature3 = vrep.simxGetObjectHandle(clientID, 'feature3', vrep.simx_opmode_oneshot_wait)
error_code, h_feature4 = vrep.simxGetObjectHandle(clientID, 'feature4', vrep.simx_opmode_oneshot_wait)
error_code, h_feature5 = vrep.simxGetObjectHandle(clientID, 'feature5', vrep.simx_opmode_oneshot_wait)
error_code, h_feature6 = vrep.simxGetObjectHandle(clientID, 'feature6', vrep.simx_opmode_oneshot_wait)
error_code, h_feature7 = vrep.simxGetObjectHandle(clientID, 'feature7', vrep.simx_opmode_oneshot_wait)
error_code, h_feature8 = vrep.simxGetObjectHandle(clientID, 'feature8', vrep.simx_opmode_oneshot_wait)
error_code, h_feature9 = vrep.simxGetObjectHandle(clientID, 'feature9', vrep.simx_opmode_oneshot_wait)
features = np.array([h_feature1, h_feature2, h_feature3, h_feature4, h_feature5, h_feature6, h_feature7, h_feature8, h_feature9])
   
# Intitiate required V_REP odometry information
error_code, simtime1 = vrep.simxGetFloatSignal(clientID, 'vrep_simtime', vrep.simx_opmode_streaming) #some error_code here for some reason
error_code, simtime1 = vrep.simxGetFloatSignal(clientID, 'vrep_simtime', vrep.simx_opmode_buffer)
error_code, theta_L1 = vrep.simxGetJointPosition(clientID, h_motor_left, vrep.simx_opmode_streaming) # error_code
error_code, theta_L1 = vrep.simxGetJointPosition(clientID, h_motor_left, vrep.simx_opmode_buffer) 
error_code, theta_R1 = vrep.simxGetJointPosition(clientID, h_motor_right, vrep.simx_opmode_streaming) # error_code
error_code, theta_R1 = vrep.simxGetJointPosition(clientID, h_motor_right, vrep.simx_opmode_buffer) 
R_L = 0.1/2 # Radius of left wheel
R_R = R_L  
B = 0.2 # Length between front and back wheels
CALIB_ODOM = 1/2.68 # odometry calibration using radius 0.5 and B = 0.2

# Initialize V-REP vision sensor information
error_code, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(clientID, h_prox_sensor, vrep.simx_opmode_streaming)    
error_code, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(clientID, h_prox_sensor, vrep.simx_opmode_buffer)    
error_code, resolution, image = vrep.simxGetVisionSensorImage(clientID, h_car_cam, 0, vrep.simx_opmode_streaming)
error_code, resolution, image = vrep.simxGetVisionSensorImage(clientID, h_car_cam, 0, vrep.simx_opmode_buffer)

# Initialize V-REP car object position
error_code, carpos = vrep.simxGetObjectPosition(clientID, h_car, -1,vrep.simx_opmode_streaming)
error_code, carpos = vrep.simxGetObjectPosition(clientID, h_car, -1,vrep.simx_opmode_buffer)
carpos = [carpos]

# Try to control robot to drive a rectangular path in separate thread
try:
    thread_robotctrl = robotctrl.ctrlRobot(clientID, h_motor_left, h_motor_right,"Thread-robotctrl")
    thread_robotctrl.start()
except:
   print("Error: unable to start thread")

# Simulation time properties
sampletime = 5e-1
totalsimtime = 20
starttime=time.time()
time1 = starttime
timevector = [[starttime]]

# Initialize number of samples during simulation for reference
ns = int(totalsimtime/sampletime)

# Intialize odometry needed information
xodom = np.zeros((ns,1))
yodom = np.zeros((ns,1))
theta = np.zeros((ns,1))
theta[0] = -np.pi/2 #start position
w = np.zeros((ns,1))
v = np.zeros((ns,1))
dtsim = np.zeros((ns,1))

# Initialize sensing
sensed_obj_true = np.zeros((ns,1))
sensed_obj_handle = np.zeros((ns,1))
sensed_obj_pos = np.zeros((ns,2))

# Initialize particles
M = 5 # Number of particles
#S = np.zeros((4,M)) # Set of particles including weights
X = np.zeros((3,M)) # Set of particles 
# observed_objects_pos = np.zeros((2,M,1)) # observed object position in time t
mu = np.zeros((2,M,1)) # features mean position related to each particle 
mu_init = np.zeros((2,M,1)) # features mean position related to each particle 
sigma = np.zeros((2,M,1)) # feature covariance related to each particle

# Initialize process noise acovariance matricies
stddev_R = 1
R = np.eye(3) * stddev_R**2

# Initialize measurement noise acovariance matricies
stddev_Q = 0.1
Q = np.eye(3) * stddev_Q

# Initialize number of observed objects
observed_objects = []

# Initialize counter
m = 0 # counter for main loop


print('...')
print('... total running time: ', totalsimtime, 's')
print('starting fastslam...')

# Main loop
while (time.time() - starttime < totalsimtime): 
    
    # Get current computer time
    time2 = time.time()
    timestep = time2 - time1
    
    # Execute fastslam iteration each sample time
    if (timestep > sampletime):

        # Get V-REP simulation time step and save execution time
        error_code, simtime2 = vrep.simxGetFloatSignal(clientID, 'vrep_simtime', vrep.simx_opmode_buffer)
        dtsim[m] = simtime2 - simtime1
        
        # Save running time
        time1 = time.time()
        timevector.append([time.time()])
        print(time.time()-starttime)
        
        # Odometry calculation, get speed and angular frequency
        error_code, theta_L2 = vrep.simxGetJointPosition(clientID, h_motor_left, vrep.simx_opmode_buffer)
        error_code, theta_R2 = vrep.simxGetJointPosition(clientID, h_motor_right, vrep.simx_opmode_buffer)     
        v[m], w[m] = vf.calculate_odometry(theta_R2-theta_R1, theta_L2-theta_L1, B, R_R, R_L, dtsim[m], CALIB_ODOM)
        error_code, simtime1 = vrep.simxGetFloatSignal(clientID, 'vrep_simtime', vrep.simx_opmode_buffer)
        error_code, theta_L1 = vrep.simxGetJointPosition(clientID, h_motor_left, vrep.simx_opmode_buffer)
        error_code, theta_R1 = vrep.simxGetJointPosition(clientID, h_motor_right, vrep.simx_opmode_buffer) 
                       
        # Extract sensor information for this time step
        error_code, detection_state, detected_point, detected_object_handle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(clientID, h_prox_sensor, vrep.simx_opmode_buffer)    
        sensed_obj_true[m] = detection_state
        if (sensed_obj_true[m]):
            sensed_obj_handle[m] = detected_object_handle
            sensed_obj_pos[m,0] = detected_point[0]
            sensed_obj_pos[m,1] = detected_point[1]
        
        # Save exact car location for reference
        error_code, carpos2 = vrep.simxGetObjectPosition(clientID, h_car, -1, vrep.simx_opmode_buffer)
        carpos.append(carpos2)
        
        # Save odometry position estimate for next car position without diffusion for reference
        xodom[m+1], yodom[m+1], theta[m+1] = vf.predict_motion_xytheta(xodom[m], yodom[m], theta[m], v[m], w[m], dtsim[m])
        
        # Start fastslam
        
        # proposal distribution for particles based on odometry and previous poses
        # Sbar = vf.predict_motion(S, v[m], w[m], R, dtsim[m])
        
        # Sample pose
        Xbar = vf.sample_pose(X, v[m], w[m], R, dtsim[m])
        
        # if feature observed
        if (detection_state):
            
            # if feature never seen before
            if (detected_object_handle not in observed_objects):
                
                print('...observed object ', detected_object_handle, 'is new...')  
                
                # Add feature to feature list
                observed_objects.append(detected_object_handle)
                
                # Save the index of the feature
                j = observed_objects.index(detected_object_handle) # get index of the already observed object
             
                # Calculate range and angle to feature related to each particle in the particle set
                # observed_objects_pos[:,:1] = np.asarray(detectedPoint[0:2]).reshape(2,1)
                observed_objects_pos = np.asarray(detectedPoint[0:2]).reshape(2,1)
                z = vf.observation_model(Xbar,observed_objects_pos,j,Q)
                
                # Initialize mean x,y position of feature based on range and angle in z
                mu_init[:,:,0] = vf.init_mean_from_z(Xbar, z)
                
                # Add to list of mean position x,y of features
                if j == 0:
                    mu = mu_init
                else:
                    mu = np.concatenate((mu,mu_init),axis=2)
                
                # calculatione observation model jacobian
                H = calculate_measurement_jacobian(Xbar,mu)
                
                # Initialize covariance
                # Sigma  = H * np.inv(Q) * H.T
                
            
            # if seen before
            else:
                j = observed_objects.index(detected_object_handle)
                print('...observed object ', detected_object_handle, 'have been seen before...')  
        # for the other features  
        else:   
            mu = mu
            sigma = sigma
            # mean and covariance the same
            
        # resample
        
        # test of ploting the course using drawnow?
        

        # Before ending loop, increment iteration 
        m = m + 1

# plot true path and map features
time.sleep(0.2)
error_code, feat1pos = vrep.simxGetObjectPosition(clientID, h_feature1, -1,vrep.simx_opmode_oneshot_wait)
error_code, feat2pos = vrep.simxGetObjectPosition(clientID, h_feature2, -1,vrep.simx_opmode_oneshot_wait)
error_code, feat3pos = vrep.simxGetObjectPosition(clientID, h_feature3, -1,vrep.simx_opmode_oneshot_wait)
error_code, feat4pos = vrep.simxGetObjectPosition(clientID, h_feature4, -1,vrep.simx_opmode_oneshot_wait)
error_code, feat4pos = vrep.simxGetObjectPosition(clientID, h_feature4, -1,vrep.simx_opmode_oneshot_wait)
error_code, feat5pos = vrep.simxGetObjectPosition(clientID, h_feature5, -1,vrep.simx_opmode_oneshot_wait)
error_code, feat6pos = vrep.simxGetObjectPosition(clientID, h_feature6, -1,vrep.simx_opmode_oneshot_wait)
error_code, feat7pos = vrep.simxGetObjectPosition(clientID, h_feature7, -1,vrep.simx_opmode_oneshot_wait)
error_code, feat8pos = vrep.simxGetObjectPosition(clientID, h_feature8, -1,vrep.simx_opmode_oneshot_wait)
error_code, feat9pos = vrep.simxGetObjectPosition(clientID, h_feature9, -1,vrep.simx_opmode_oneshot_wait)
featpos = [feat1pos, feat2pos, feat3pos, feat4pos, feat5pos,feat6pos, feat7pos, feat8pos, feat9pos]
featpos = np.asarray(featpos)
t = np.asarray(timevector)
t = t - t[0]
carpos = np.asarray(carpos)
xtrue = carpos[:,0]
ytrue = carpos[:,1]
plt.plot(xtrue,ytrue,xodom,yodom,featpos[0,0],featpos[0,1],'*',featpos[1,0],featpos[1,1],'*',featpos[2,0],featpos[2,1],'*',featpos[3,0],featpos[3,1],'*',featpos[4,0],featpos[4,1],'*',featpos[5,0],featpos[5,1],'*',featpos[6,0],featpos[6,1],'*',featpos[7,0],featpos[7,1],'*',featpos[8,0],featpos[8,1],'*')


"""
continue with fastslam algorithm for feature never seen before
think about time vector for particles or how to plot each time sample, otherwise difficult to problem solve
"""

"""
OLD STUFF
"""
# Odometry
#dtheta_L = np.zeros((ns,1))
#dtheta_R = np.zeros((ns,1))
#w_R = np.zeros((ns,1))
#w_L = np.zeros((ns,1))
#        dtheta_R[m] = theta_R2 - theta_R1
#        dtheta_L[m] = theta_L2 - theta_L1
#        w_R[m] = dtheta_R[m] / dtsim[m]
#        w_L[m] = dtheta_L[m] / dtsim[m]
#        w[m] = CALIB_ODOM * (w_R[m]*R_R - w_L[m]*R_L) / B
#        v[m] = (w_R[m]*R_R + w_L[m]*R_L) / 2


        # save odometry information for plotting (motion model from thrun p.127, non-working)
#        xodom2[m+1] = xodom2[m] - (v[m]/(w[m] + 1e-9)) * np.sin(theta[m]) + (v[m]/(w[m] + 1e-9)) * np.sin(theta[m] + w[m]*dtsim[m])
#        yodom2[m+1] = yodom2[m] + (v[m]/(w[m] + 1e-9)) * np.cos(theta[m]) - (v[m]/(w[m] + 1e-9)) * np.cos(theta[m] + w[m]*dtsim[m])
               
# Get camera image and plot it
#error_code, resolution, car_image = vrep.simxGetVisionSensorImage(clientID, h_car_cam, 0, vrep.simx_opmode_buffer)
#time.sleep(0.2)
#im = np.array(car_image, dtype = np.uint8)
#im.resize([resolution[0],resolution[1],3])
#im.shape
#plt.imshow(im, origin='lower')        
        
## Test with multiple objects sensed  - send string signal with floats
#error_code, signalValue = vrep.simxGetStringSignal(clientID,"myStringSignalName",vrep.simx_opmode_streaming)
#error_code, signalValue = vrep.simxGetStringSignal(clientID,"myStringSignalName",vrep.simx_opmode_buffer)
#vrep.simxUnpackFloats(signalValue)
## Test with multiple objects - send sensed data (NOT WORKING!?)
#error_code, signalValue2 = vrep.simxGetStringSignal(clientID,"testString",vrep.simx_opmode_streaming)
#error_code, signalValue2 = vrep.simxGetStringSignal(clientID,"testString",vrep.simx_opmode_buffer)
#vrep.simxUnpackFloats(signalValue2)        