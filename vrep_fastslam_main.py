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
import matplotlib.animation as animation # for animated plotting
import matplotlib.patches as mpatches # used for legend
import itertools # for reducing a 2d list to 1d list

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


# close all windows 
plt.close("all")
# Stop-Start simulation of robot
#error_code=vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot)
#time.sleep(1) # wait for sim to stop
#error_code = vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot)

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

# Initialize particles (landmark, particles, rows, cols)
M = 5 # Number of particles
Xstart = np.array([0, 0, -np.pi/2]) # Assumed particle start position
X = np.repeat(Xstart[:, np.newaxis], M, axis=1) # Set of particles 
particles_xpos = [X[0,:].tolist()]
particles_ypos = [X[1,:].tolist()]
mu = np.zeros((1,2,M)) # features mean position related to each particle 1X2XM
mu_init = np.zeros((1,2,M)) # features mean position related to each particle 1X2XM
mu_new = np.zeros((1,2,M))
Sigma = np.zeros((1,M,3,3)) # feature position covariance related to each particle and feature
Sigma_init = np.zeros((1,M,3,3)) # feature position covariance related to each particle and feature
Sigma_new = np.zeros((1,M,3,3))

# Initialize process noise acovariance matricies
stddev_R = 1
R = np.eye(3) * stddev_R**2

# Initialize measurement noise acovariance matricies
stddev_Qt = 0.1
stddev_range = 0.1
stddev_bearing = 0.01
Qt = np.eye(2) * stddev_Qt
Qt = np.array([[stddev_range, 0],[0, stddev_bearing]])
QtS = np.repeat(Qt[np.newaxis, :, :], M, axis=0)
QtSinv = np.linalg.inv(QtS)

# Initialize number of observed objects
observed_objects = []

# Initialize identify matrix
I = np.repeat(np.eye(3)[np.newaxis, :, : ], M, axis=0)

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
        Xbar = vf.sample_pose(X, v[m], w[m], R, dtsim[m]) # 3XM ([x,y,theta]'XM)
        
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
                observed_objects_pos = np.asarray(detectedPoint[0:2]).reshape(2,1) #2X1
                z = vf.observation_model(Xbar,observed_objects_pos,j,Qt) # 2XM
                
                # Initialize mean x,y position of feature based on range and angle in z
                mu_init[0,:,:] = vf.init_mean_from_z(Xbar, z) # 1X2XM
                
                # Add to list of mean position x,y of features
                if j == 0:
                    mu = mu_init # 2XMX1
                else:
                    mu = np.concatenate((mu,mu_init),axis=0) # NX2XM
                
                # calculatione observation model jacobian 
                H = vf.calculate_measurement_jacobian(Xbar,mu,j) # # MX2X3. Only for this feature 
                
                # Make a transpose along the 3x3 dimension for each particle
                H_T = np.transpose(H,(1,0,2)) # MX3X2
                
                # Inverse of jacobian of measurement model 
                # https://stackoverflow.com/questions/41850712/compute-inverse-of-2d-arrays-along-the-third-axis-in-a-3d-array-without-loops
                # Hinv = np.linalg.inv(H.T).T
                
                # Make a transpose along the 2x3 dimension for each particle M
                H_T = np.transpose(H,(0,2,1)) # MX3X2
                
                # Invert Q measurement noise matrix
                # Qinv = np.linalg.inv(Qt) # 3X3
                
                # Scale the Q matrix for each particle
                # indexing with np.newaxis inserts a new 3rd dimension, which we then repeat the
                # array along, (you can achieve the same effect by indexing with None, see below)
                # QinvS = np.repeat(Qinv[:, :, np.newaxis], M, axis=2) # 3X3XM

                # Initialize covariance 
                #Sigma_init[0,:,:,:]  = np.linalg.inv(H_T @ QtSinv @ H) # 1XMX3X3
                Sigma_init[0,:,:,:]  = (H_T @ QtSinv @ H)
                
                
                # Add to list of sigma covariance of features
                if j == 0:
                    Sigma = Sigma_init #MX3X3
                else:
                    Sigma = np.concatenate((Sigma,Sigma_init),axis=0) # NXMX3X3              
                
                # default importance weights, should be equal to one
                weights = 1/M*np.ones((1,M)) # 1XM
            
            # else if feature has been seen before
            else:
                j = observed_objects.index(detected_object_handle)
                print('...observed object ', detected_object_handle, 'have been seen before...')  
                
                # measurement prediction based on particle set X and mean feature position mu
                zhat = vf.observation_model_zhat(Xbar,mu,j,Qt) # 2XM
                
                # calculate jacobian
                H = vf.calculate_measurement_jacobian(X,mu,j) # MX2X3
                
                # Make a transpose along the 3x3 dimension for each particle
                H_T = np.transpose(H,(0,2,1)) # MX3X2     
                
                # Measurement covariance (not the same as Qt measurement covariance noise)
                Q = H @ Sigma[j,:,:,:] @ H_T + QtS # MX2X2
                
                # Inverse of measurement covariance
                Qinv = np.linalg.inv(Q) # MX2X2
                
                # Calculate Kalman gain
                K = Sigma[j,:,:,:] @ H_T @ Qinv # MX3X2
                
                # innovation 2XM
                nu = (z-zhat) # 2XM
                
                # correct angle innovation to be within -pi and pi
                nu[1,:] = ((nu[1,:] + np.pi) % (2*np.pi)) - np.pi
               
                # Add extra row to measurement error for Kalman gain multiplication
                #zerror = np.concatenate((z,np.zeros((1,M))),axis=0) # 3XM, but I want it MX3X1
                nu = np.transpose(nu.reshape(2,5,1),(1,0,2)) # MX2X1
                
                # update mean. Is this really correct??? mixing coordinates? features x,y. nu is in r,theta. Kalman gain ???.
                mu[j,:,:] = mu[j,:,:] + (K @ nu).T[:,:2,:] # NX2XM # think about changing to NXMX2X1
                
                # update covariance
                Sigma[j,:,:,:] = (I - K@H) @ Sigma[j,:,:,:]# NXMX3X3
                
                # importance factor amplitude. Correct with det(Q)? consider checking H
                ata = 1/np.sqrt(np.pi*2*np.linalg.det(Q)) # 1XM
                
                # Transpose innovation nu
                nuT = np.transpose(nu,(0,2,1))
                
                # Mahalonobis distance
                D = nuT @ Qinv @ nu # MX1X1
                
                # importance factors
                weights_not_normalized = ata*np.exp(-0.5*D.reshape(1,5)) # 1XM
                
                # normalize the weights
                weightsum = np.sum(weights_not_normalized)
                weights = weights_not_normalized / weightsum
                
                
                
        # if no feature was detected or for the features that were not detected 
        else:  
            # mean and covariance the same as earlier time step
            mu = mu
            Sigma = Sigma
                       
        # Resampling  
        
        # Re-initialize the particles
        X = np.zeros((3,M))
             
        # systematic resampling
        cdf = np.cumsum(weights)
        rval = np.asscalar(np.random.rand(1)) / M # uniform distributed random value between 0 and 1/M
        
        # draw a sample with a proportional probability to the weights
        for n in range(0,M-1):
            
            # find first value equal to or above the random selected value
            select = np.searchsorted(cdf,rval,'right') 
            
            # Save this particle for next iteration
            X[:,n] = Xbar[:,select]
            
            # increment the randomly selected value
            rval = rval + 1/M
            
        # Re-initialize the weights
        weights = 1/M*np.ones((1,M))
            
        
        # save data for plotting
        particles_xpos.append(X[0,:].tolist())
        particles_ypos.append(X[1,:].tolist())

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

fig, ax = plt.subplots()
csfont = {'fontname':'sans-serif'} # Verdana, Arial
hfont = {'fontname':'sans-serif'}
ax.grid(True)
plt.title('Simulated world',**csfont)
plt.ylabel('y',**hfont)
plt.xlabel('x',**hfont)
xdata, ydata = [], []
xdata2, ydata2 = [], []
xdata3, ydata3 = [], []
line, = plt.plot([], [], 'ro', animated=True)
line2, = plt.plot([], [], 'g*', animated=True)
line3, = plt.plot([], [], 'b+', animated=True)



#b : blue.
#g : green.
#r : red.
#c : cyan.
#m : magenta.
#y : yellow.
#k : black.
#w : white.
red_patch = mpatches.Patch(color='red', label='exact car position')
green_patch = mpatches.Patch(color='green', label='odometry information')
blue_patch = mpatches.Patch(color='blue', label='particles')
plt.legend(handles=[red_patch, green_patch , blue_patch])

#particles_xpos1d = list(itertools.chain.from_iterable(particles_xpos))
#particles_ypos1d = list(itertools.chain.from_iterable(particles_xpos))

# set the axis of the plot
def init():
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    return line,

# data to use
def update(frame):
    xdata.append(xtrue[frame])
    ydata.append(ytrue[frame])
    xdata2.append(xodom[frame])
    ydata2.append(yodom[frame]) 
    xdata3.append(particles_xpos[frame])
    ydata3.append(particles_ypos[frame]) 
    line.set_data(xdata, ydata)
    line2.set_data(xdata2, ydata2)   
    line3.set_data(xdata3, ydata3)
    return line, line2, line3

anim = animation.FuncAnimation(fig, update, interval=100, frames=ns, init_func=init, blit=True)

# to save the file as an animation, one needs ffmpeg. Anaconda: conda install -c conda-forge ffmpeg
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
anim.save('basic_animation.mp4', writer=writer)

# plot landmarks
plt.plot(featpos[0,0],featpos[0,1],'m*',featpos[1,0],featpos[1,1],'m*',featpos[2,0],featpos[2,1],'m*',featpos[3,0],featpos[3,1],'m*',featpos[4,0],featpos[4,1],'m*',featpos[5,0],featpos[5,1],'m*',featpos[6,0],featpos[6,1],'m*',featpos[7,0],featpos[7,1],'m*',featpos[8,0],featpos[8,1],'m*')

plt.show()



#from datetime import datetime
#from matplotlib import pyplot
#from matplotlib.animation import FuncAnimation
#from random import randrange
#
#x_data, y_data = [], []
#
#figure = pyplot.figure()
#line, = pyplot.plot_date(x_data, y_data, '-')
#
#def update(frame):
#    x_data.append(datetime.now())
#    y_data.append(randrange(0, 100))
#    line.set_data(x_data, y_data)
#    figure.gca().relim()
#    figure.gca().autoscale_view()
#    return line,
#
#animation = FuncAnimation(figure, update, interval=200)
#
#pyplot.show()
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