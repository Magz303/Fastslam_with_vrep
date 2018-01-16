
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 00:35:16 2018

@author: Magnus Tarle

Containts functions and simplified test functions for vrep_fastslam
"""
import numpy as np
import matplotlib.patches as mpatches # used for legend, ellipses and rectangles

def feature_pos_and_cov(X,mu,Sigma,k):
    mu_mean = np.mean(mu,axis=2)
    X_mean = np.mean(X,axis=1)
    Sigma_mean = np.mean(Sigma,axis=1)

    r = np.sqrt((mu_mean[k,0] - X_mean[0])**2 +(mu_mean[k,1] - X_mean[1])**2) # range to feature for each particle
#    theta = np.arctan2(mu_mean[k,1]-X_mean[1] , mu_mean[k,0]-X_mean[0]) - X_mean[2] # angle to observed feature for each particle
    theta = np.arctan2(mu_mean[k,0]-X_mean[0] , mu_mean[k,1]-X_mean[1])
    theta_lim = ((theta + np.pi) % (2*np.pi)) - np.pi # limit angle between pi and -pi    
    
#    xfeat_world = X_mean[0] + r * np.cos(X_mean[2]+theta) # X position of particle plus x-distance to feature related to the particle
#    yfeat_world = X_mean[1] + r * np.sin(X_mean[2]+theta) # the same for Y
    xfeat_world = X_mean[0] + r * np.sin(theta + X_mean[2]) # X position of particle plus x-distance to feature related to the particle
    yfeat_world = X_mean[1] + r * np.cos(theta + X_mean[2])
    return xfeat_world, yfeat_world

# USED IN SIM!
def add_arrow_object(iter,carpos,featpos,feature_index):
    carpos = np.asarray(carpos)
    feature_index = feature_index -1
    a = mpatches.Arrow(x=carpos[iter,0], y=carpos[iter,1], dx=featpos[feature_index,0]-carpos[iter,0],dy=featpos[feature_index,1]-carpos[iter,1], width = 0.2, color = 'red')
    return a
    
# USED IN SIM!
def id_feature(features,sensed_object):
    # This function retrieves the index of the observed object handle in vrep
    itemindex = np.where(features==sensed_object)
    return int(itemindex[0])+1

# USED IN SIM!
def init_mean_from_z(X, z): # see p.320
    # This function should initialize the mean location mu of the feature in world coordinates
    # based on a range and angle z related to the particle set X
    # The bearing lies in the interval [-pi,pi)
    # Inputs:    
    #           X           3XM previous particle set representing the states and the weights [x,y,theta]
    #           z           2XM observation function for range-bearing measurement, [r, theta]'    '
    # Outputs:  
    #           mu_init     2XM position of the feature related to each particle
    M = np.size(X[0,:])     # Number of particles in particle set  
    mu_init = np.zeros((2,M))
#    mu_init[0,:] = X[0,:] + z[0,:] * np.cos(X[2,:]+z[1,:]) # X position of particle plus x-distance to feature related to the particle
#    mu_init[1,:] = X[1,:] + z[0,:] * np.sin(X[2,:]+z[1,:]) # the same for Y
    mu_init[0,:] = X[0,:] + z[0,:] * np.sin(X[2,:]+z[1,:]) # X position of particle plus x-distance to feature related to the particle
    mu_init[1,:] = X[1,:] + z[0,:] * np.cos(X[2,:]+z[1,:]) # the same for Y    
    return mu_init


def test_init_mean():
    xytheta_dim = 3
    particles_dim = 7
    X = np.zeros((xytheta_dim, particles_dim))
    j= 1 # observed feature
    N = 9 # number of features
    W = np.ones((2,N))*1
    z = observation(X, W, j)
    mu_init1 = init_mean_from_z(X, z)
    # mu_init2 = init_mean_from_xy(X, W[:,:1])
    print('observation model z', z)    
    print('init_mean1', mu_init1)
#    print('init_mean1: ', mu_init1.shape)  
    # print('init_mean2', mu_init2)
#    print('init_mean2: ', mu_init2.shape)

#test_init_mean()
    
# USED IN SIM!    
def calculate_measurement_jacobian(X,mu,j): 
    # This function calculates the jacobian of the observation model for a given
    # feature and the particle set. The Jacobian is used
    # later to obtain the measurement covariance matrix Sigma = HxQxH.T  
    # for a given feature and the particle set.
    # The derivative is based on the feature x,y and not the robot itself which has x,y and theta
    # Inputs:
    #           X(t)    3XM estimated states [x,y,theta]'   
    #           mu(t)   NX2XM observed mean position of features [x,y]'       
    # Outputs:  
    #           H       MX2X3 H is the Jacobian of h corresponding to any observation evaluated at mu_bar t
    
    # Get variables
    M = np.size(X[0,:])     # Number of particles in particle set    
    mux = mu[j,0,:] # Extract one feature j and create a matrix shape for creating a distance calculation in x for all particles
    muy = mu[j,1,:] # Extract one feature j and create a matrix shape for creating a distance calculation in y for all particles
    Xx = X[0,:] # Extract the x position of all particles
    Xy = X[1,:] # Extract the y position of all particles
    q = (mux - Xx)**2 +(muy - Xy)**2 # 

    # linearization of observation model z = [r theta] where range is first
    # (range = np.sqrt((mux-Xx)**2 + (mux-Xy)**2) )
    # (theta = np.arctan2(muy-Xy,mux-Xx) - Xtheta )
    
    # range linearization
    dhr_dmux = (mux-Xx) / np.sqrt(q)
    dhr_dmuy = (muy-Xy) / np.sqrt(q)
    
    # angle linearization
    dhtheta_dmux = -(mux-Xx) / q #1XM
    dhtheta_dmuy = (muy-Xy) / q
    
    dhr_dmux2 = dhr_dmux.reshape(M,1,1)
    dhr_dmuy2 = dhr_dmuy.reshape(M,1,1)
#    dhr_dtheta2 = dhr_dtheta.reshape(M,1,1)    
    dhtheta_dmux2 = dhtheta_dmux.reshape(M,1,1)
    dhtheta_dmuy2 = dhtheta_dmuy.reshape(M,1,1)
#    dhtheta_dtheta = dhtheta_dtheta.reshape(M,1,1)
#    H1 = np.concatenate((dhr_dx2,dhr_dy2,dhr_dtheta2),axis=2)
#    H2 = np.concatenate((dhtheta_dx2,dhtheta_dy2,dhtheta_dtheta),axis=2)
    H1 = np.concatenate((dhr_dmux2,dhr_dmuy2),axis=2)
    H2 = np.concatenate((dhtheta_dmux2,dhtheta_dmuy2),axis=2)
    H = np.concatenate((H1,H2), axis=1) # MX2X2
    return H


# USED IN SIM!    
def observation_model_zhat(X,mu,j,Q): # maybe remove Q and j here....
    # This function implements the observation model and calculates the range and angle zhat for 
    # a given feature j in mean [x,y] coordinates mu (in relation to each particle) and particle set X
    # Note: The bearing theta lies in the interval [-pi,pi) in relation to the particle set X
    # Inputs:
    #           X           3XM previous particle set representing the states and the weights [x,y,theta]'
    #           mu          NX2XM coordinates of the features in x,y in tth time 
    #           j           1X1 index of the feature being matched to the measurement observation
    #           Q           2X2 measurement covariance noise   
    # Outputs:  
    #           z           2XM observation function for range-bearing measurement, [r, theta]'
    M = np.size(X[0,:])     # Number of particles in particle set    
    Featx = mu[j,0,:] # Extract one feature j for creating a distance calculation in x for all particles
    Featy = mu[j,1,:] # Extract one feature j and create a matrix shape for creating a distance calculation in y for all particles
    Xx = X[0,:] # Extract the x position of all particles
    Xy = X[1,:] # Extract the y position of all particles
    Xtheta = X[2,:] # Extract the theta angle of all particles
    ra = np.sqrt((Featx - Xx)**2 +(Featy - Xy)**2) # range to feature for each particle
    ra = ra.reshape(1,M)
    theta = np.arctan2(Featy-Xy,Featx-Xx) - Xtheta # angle to observed feature for each particle
    theta_lim = ((theta + np.pi) % (2*np.pi)) - np.pi # limit angle between pi and -pi
    theta_lim = theta_lim.reshape(1,M)
    zmeas = np.concatenate((ra, theta_lim), axis = 0)
    # Add diffusion
    rtheta_stddev2 = np.diag(np.sqrt(Q)) # obtain standard deviation square of process noise (1-dimensional array)
    diffusion_normal = np.random.standard_normal((2,M))  # Normal distribution with standard deviation 1 
    diffusion = diffusion_normal * rtheta_stddev2.reshape(2,1) # Normal distribution with standard deviation according to process noise covariance R (reshape sigma to get 3 rows and 1 column for later matrix inner product multiplication)
    z = zmeas + diffusion[:2,:] # estimated states = old states + motion + diffusion    
    return z    

def z_from_detectection(X,observed_objects_pos): # maybe remove Q and j here....
    # This function implements the observation model and calculates the range and angle z for 
    # a given feature j in [x,y] coordinates W and particle set X
    # Note: The bearing theta lies in the interval [-pi,pi) in relation to the particle set X
    # Inputs:
    #           X           3XM previous particle set representing the states and the weights [x,y,theta]'
    #           W           2XN coordinates of the features in x,y in tth time 
    #           j           1X1 index of the feature being matched to the measurement observation
    #           Q           2X2 measurement covariance noise   
    # Outputs:  
    #           z           2XM observation function for range-bearing measurement, [r, theta]'
    M = np.size(X[0,:])     # Number of particles in particle set    
    Featx = np.ones((1,M)) * observed_objects_pos[0,0] # Extract one feature j and create a matrix shape for creating a distance calculation in x for all particles
    Featy = np.ones((1,M)) * observed_objects_pos[1,0] # Extract one feature j and create a matrix shape for creating a distance calculation in y for all particles
    r = np.sqrt((Featx)**2 +(Featy)**2) # range to feature for each particle
    theta = np.arctan2(Featx,Featy) # angle to observed feature for each particle
    theta_lim = ((theta + np.pi) % (2*np.pi)) - np.pi # limit angle between pi and -pi
    zmeas = np.concatenate((r, theta_lim), axis = 0)
    z = zmeas# + diffusion[:2,:] # estimated states = old states + motion + diffusion    
    return zmeas

def observation_model(X,W,j,Q): # maybe remove Q and j here....
    # This function implements the observation model and calculates the range and angle z for 
    # a given feature j in [x,y] coordinates W and particle set X
    # Note: The bearing theta lies in the interval [-pi,pi) in relation to the particle set X
    # Inputs:
    #           X           3XM previous particle set representing the states and the weights [x,y,theta]'
    #           W           2XN coordinates of the features in x,y in tth time 
    #           j           1X1 index of the feature being matched to the measurement observation
    #           Q           2X2 measurement covariance noise   
    # Outputs:  
    #           z           2XM observation function for range-bearing measurement, [r, theta]'
    M = np.size(X[0,:])     # Number of particles in particle set    
    Featx = np.ones((1,M)) * W[0,0] # Extract one feature j and create a matrix shape for creating a distance calculation in x for all particles
    Featy = np.ones((1,M)) * W[1,0] # Extract one feature j and create a matrix shape for creating a distance calculation in y for all particles
    Xx = X[0,:] # Extract the x position of all particles
    Xy = X[1,:] # Extract the y position of all particles
    Xtheta = X[2,:] # Extract the theta angle of all particles
    r = np.sqrt((Featx - Xx)**2 +(Featy - Xy)**2) # range to feature for each particle
    theta = np.arctan2(Featy-Xy,Featx-Xx) - Xtheta # angle to observed feature for each particle
    theta_lim = ((theta + np.pi) % (2*np.pi)) - np.pi # limit angle between pi and -pi
    zmeas = np.concatenate((r, theta_lim), axis = 0)
    # Add diffusion
    rtheta_stddev2 = np.diag(np.sqrt(Q)) # obtain standard deviation square of process noise (1-dimensional array)
    diffusion_normal = np.random.standard_normal((2,M))  # Normal distribution with standard deviation 1 
    diffusion = diffusion_normal * rtheta_stddev2.reshape(2,1) # Normal distribution with standard deviation according to process noise covariance R (reshape sigma to get 3 rows and 1 column for later matrix inner product multiplication)
    z = zmeas# + diffusion[:2,:] # estimated states = old states + motion + diffusion    
    return z

def test_observation_model():
    xytheta_dim = 3
    particles_dim = 7
    X = np.ones((xytheta_dim, particles_dim))
    j= 1 # observed feature
    N = 9 # number of features
    W = np.ones((2,N))*2
    Q = np.eye(2)*0.01
    z = observation_model(X, W, j, Q)
    print('observation model z', z)
    print('z dimension: ', z.shape)  

#test_observation_model()

# USED IN SIM!
def sample_pose(X, v, w, R, delta_t):
    # This function perform the prediction step / proposal distribution for the particle set X
    # based on the odometry speed and angular frequency as well as the process noise R
    # Inputs:
    #           X(t-1)              3XM previous time particle set representing the states [x,y,theta]'
    #           v(t)                1X1 translational velocity of the robot in the tth time step
    #           w(t)                1X1 angular velocity of the robot in the tth time step
    #           R                   3X3 process noise covariance matrix of x,y and theta
    #           delta_t             1X1 discrete time step between time t and t-1
    # Outputs:
    #           Xbar(t)             3XM prediction of the new states [x,y,thetha]
    M = np.size(X[0,:])             # Number of particles in particle set
    xytheta_dim = 3 # Number of states of each particle 
    Xbar = np.zeros((3,M))
    theta_prev = X[2,:]  # theta angle at previous time step
#    xy_predmotion = delta_t * np.array([ v * np.cos(theta_prev), v * np.sin(theta_prev) ]) # motion model on x and y for each particle  
    xy_predmotion = delta_t * np.array([ -v * np.sin(theta_prev), v * np.cos(theta_prev) ]) # motion model on x and y for each particle  
    theta_predmotion = delta_t * w * np.ones((1,M))  #motion model for the angle of each particle 
    xytheta_predmotion = np.concatenate((xy_predmotion, theta_predmotion), axis = 0) # combine the complete motion model x,y and theta into one matrix 
    xytheta_sigma = np.diag(np.sqrt(R)) # obtain standard deviation of process noise (1-dimensional array)
    diffusion_normal = np.random.standard_normal((xytheta_dim,M))  # Normal distribution with standard deviation 1 
    diffusion = diffusion_normal * xytheta_sigma.reshape(3,1) # Normal distribution with standard deviation according to process noise covariance R (reshape sigma to get 3 rows and 1 column for later matrix inner product multiplication)
    Xbar = X + xytheta_predmotion + diffusion # estimated states = old states + motion + diffusion
    return Xbar

def test_sample_pose():
    # Inputs:
    #           X(t-1)              3XM previous time particle set representing the states [x,y,theta]'
    #           v(t)                1X1 translational velocity of the robot in the tth time step
    #           w(t)                1X1 angular velocity of the robot in the tth time step
    #           R                   3X3 process noise covariance matrix of x,y and theta
    #           delta_t             1X1 discrete time step between time t and t-1
    # Outputs:
    #           Xbar(t)             3XM prediction of the new states [x,y,thetha]
    xyw = 3
    particles = 7
    X = np.ones((xyw, particles))
    v = 1
    w = 0.5
    R = np.eye(3)
    delta_t = 0.1
    Xbar = sample_pose(X, v, w, R, delta_t)
    print('Xbar', Xbar)
    print('Xbar dimension: ', Xbar.shape)

#test_sample_pose()    
    
# USED IN SIM!
def calculate_odometry(delta_angle_R, delta_angle_L, B, R_R, R_L, delta_t, CALIB_ODOM):
    #  This function calculates translational speed and angular frequency w of the robot provided
    # from the odometry information of the robot wheel motors
    #  Inputs:
    #       delta_angle_R(t): 1X1 angle difference for the left wheel in the tth time step
    #       delta_angle_L(t): 1X1 angle difference for the right wheel in the tth time step
    #       B:                1X1 wheel base (distance between the contact points of the wheels, front and back)
    #       R_L:              1X1 radius of the left wheels
    #       R_R:              1X1 radius of the right wheels
    #       delta_t:          1X1 previous state in [x, y, theta]'
    #       CALIB_ODOM:       1X1 calibration constant for odometry slip etc
    #% Outputs:
    #%      v(t):             1X1 translational velocity of the robot in the tth time step
    #%      w(t):             1X1 angular velocity of the robot in the tth time step
    w_R = delta_angle_R / delta_t
    w_L = delta_angle_L / delta_t
    w = (w_R*R_R - w_L*R_L) / B * CALIB_ODOM
    v = (w_R*R_R + w_L*R_L) / 2
    return v, w

    
def test_calculate_odometry():
    #  Inputs:
    #       delta_angle_R(t): 1X1 angle difference for the left wheel in the tth time step
    #       delta_angle_L(t): 1X1 angle difference for the right wheel in the tth time step
    #       B:                1X1 wheel base (distance between the contact points of the wheels, front and back)
    #       R_L:              1X1 radius of the left wheels
    #       R_R:              1X1 radius of the right wheels
    #       delta_t:          1X1 previous state in [x, y, theta]'
    #       CALIB_ODOM:       1X1 calibration constant for odometry slip etc
    #% Outputs:
    #%      v(t):             1X1 translational velocity of the robot in the tth time step
    #%      w(t):             1X1 angular velocity of the robot in the tth time step
    R_L = 0.1/2 # Radius of left wheel
    R_R = R_L  
    B = 0.2 # Length between front and back wheels
    CLBRTE_ODMTRY = 1/2.68 # odometry calibration using radius 0.5 and B = 0.2
    delta_t = 0.1
    angle_diff_L = np.pi
    angle_diff_R = np.pi*1.1
    v, w = calculate_odometry(angle_diff_R, angle_diff_L, B, R_R, R_L, delta_t, CLBRTE_ODMTRY)
    print('velocity is', v)
    print('angular velocity is', w)
    
# USED IN SIM!
def predict_motion_xytheta(x, y, theta, v, w, delta_t):
    # This function performs a prediction step without diffusion and without weights, i.e.
    # estimates the next time increment position of the robot based on the robot position
    # and translational velocity and angular frequency
    # Inputs:
    #           x(t-1)             1X1 previous state x-direction
    #           y(t-1)             1X1 previous state y-direction
    #           theta(t-1)         1X1 previous state theta-angle    
    #           v(t)                1X1 translational velocity of the robot in the tth time step
    #           w(t)                1X1 angular velocity of the robot in the tth time step
    #           delta_t             1X1 discrete time step between time t and t-1
    # Outputs: 
    #           x_predmotion(t)                1X1 prediction of the new state x-direction
    #           y_predmotion(t)                1X1 prediction of the new state y-direction
    #           theta_predmotion(t)            1X1 prediction of the new state theta-angle    
    x_predmotion = x - v*delta_t*np.sin(theta)
    y_predmotion = y + v*delta_t*np.cos(theta)
    theta_predmotion = (((theta + w*delta_t) + np.pi) % (2*np.pi)) - np.pi # set angle between -pi and pi
    return x_predmotion, y_predmotion, theta_predmotion
    
# test_calculate_odometry()
# test_predict_motion()
# test_observation_model_particle()
    

########
    # Not used
    
#def observation_model_particle(S,W,j):
#    # This function implements the observation model, might need modification as S contains landmarks as well!
#    # The bearing lies in the interval [-pi,pi)
#    # Inputs:
#    #           S           4XM previous particle set representing the states and the weights [x,y,theta, weights]'
#    #           W           2XN coordinates of the features in x,y in tth time 
#    #           j           1X1 index of the feature being matched to the measurement observation
#    # Outputs:  
#    #           h           2XM observation function for range-bearing measurement, [r, theta]'
#    M = np.size(S[0,:])     # Number of particles in particle set    
#    Featx = np.ones((1,M)) * W[0,j] # Extract one feature j and create a matrix shape for creating a distance calculation in x for all particles
#    Featy = np.ones((1,M)) * W[1,j] # Extract one feature j and create a matrix shape for creating a distance calculation in y for all particles
#    Sx = S[0,:] # Extract the x position of all particles
#    Sy = S[1,:] # Extract the y position of all particles
#    Stheta = S[2,:] # Extract the theta angle of all particles
#    r = np.sqrt((Featx - Sx)**2 +(Featy - Sy)**2) # range to feature for each particle
#    theta = np.arctan2(Featy-Sy,Featx-Sx) - Stheta # angle to observed feature for each particle
#    theta_lim = ((theta + np.pi) % (2*np.pi)) - np.pi # limit angle between pi and -pi
#    h = np.concatenate((r, theta_lim), axis = 0)
#    return h
#
#def test_observation_model_particle():
#    # Inputs:
#    #           S           4XM previous particle set representing the states and the weights [x,y,theta, weights]'
#    #           W           2XN coordinates of the features in x,y in tth time 
#    #           j           1X1 index of the feature being matched to the measurement observation
#    # Outputs:  
#    #           h           2XM observation function for range-bearing measurement, [r, theta]'
#    xytheta_dim = 3
#    weights_dim = 1
#    particles_dim = 7
#    S = np.ones((xytheta_dim + weights_dim, particles_dim))
#    j= 1 # observed feature
#    N = 9 # number of features
#    W = np.ones((2,N))*2
#    h = observation_model_particle(S, W, j)
#    print('observation model h', h)
#    print('h dimension: ', h.shape)  
    
#def predict_motion(S, v, w, R, delta_t):
#    # This function perform the prediction step / proposal distribution
#    # Inputs:
#    #           S(t-1)              4XM previous time particle set representing the states and weights [x,y,theta, weights]'
#    #           v(t)                1X1 translational velocity of the robot in the tth time step
#    #           w(t)                1X1 angular velocity of the robot in the tth time step
#    #           R                   3X3 process noise covariance matrix of x,y and theta
#    #           delta_t             1X1 discrete time step between time t and t-1
#    # Outputs:
#    #           Sbar(t)             4XM prediction of the new states and weights [x,y,thetha,weights]
#    M = np.size(S[0,:])             # Number of particles in particle set
#    xytheta_dim = 3 # Number of states of each particle 
#    Sbar = np.zeros((4,M))
#    theta_prev = S[2,:]  # theta angle at previous time step
#    xy_predmotion = delta_t * np.array([ v * np.cos(theta_prev), v * np.sin(theta_prev) ]) # motion model on x and y for each particle  
#    theta_predmotion = delta_t * w * np.ones((1,M))  #motion model for the angle of each particle 
#    xytheta_predmotion = np.concatenate((xy_predmotion, theta_predmotion), axis = 0) # combine the complete motion model x,y and theta into one matrix 
#    xytheta_sigma = np.diag(np.sqrt(R)) # obtain standard deviation of process noise (1-dimensional array)
#    diffusion_normal = np.random.standard_normal((xytheta_dim,M))  # Normal distribution with standard deviation 1 
#    diffusion = diffusion_normal * xytheta_sigma.reshape(3,1) # Normal distribution with standard deviation according to process noise covariance R (reshape sigma to get 3 rows and 1 column for later matrix inner product multiplication)
#    Sbar[:3,:] = S[:3,:] + xytheta_predmotion + diffusion # estimated states = old states + motion + diffusion
#    Sbar[-1,:] = S[-1,:] # keep weights as is
#    return Sbar
#
#def test_predict_motion():
#    # Inputs:
#    #           S(t-1)              4XM previous time particle set representing the states and weights [x,y,theta, weights]'
#    #           v(t)                1X1 translational velocity of the robot in the tth time step
#    #           w(t)                1X1 angular velocity of the robot in the tth time step
#    #           R                   3X3 process noise covariance matrix of x,y and theta
#    #           delta_t             1X1 discrete time step between time t and t-1
#    # Outputs:
#    #           Sbar(t)            4XM prediction of the new states and weights [x,y,thetha,weights]
#    xyw = 3
#    weights = 1
#    particles = 7
#    S = np.ones((xyw + weights, particles))
#    v = 1
#    w = 0.5
#    R = np.eye(3)
#    delta_t = 0.1
#    Sbar = predict_motion(S, v, w, R, delta_t)
#    print('Sbar', Sbar)
#    print('Sbar dimension: ', Sbar.shape)


#def init_mean_from_xy(X, xy):
#    # This function should initialize the mean location of the feature in world coordinates
#    # The bearing lies in the interval [-pi,pi)
#    # Inputs:    
#    #           X           3XM previous particle set representing the states and the weights [x,y,theta]
#    #           xy          2XM observation function for xy measurement, [x, y]'    '
#    # Outputs:  
#    #           mu_init     2XM position of the feature related to each particle
#    M = np.size(X[0,:])     # Number of particles in particle set  
#    mu_init = np.zeros((2,M))
#    mu_init[0,:] = X[0,:] + xy[0,:] # X position of particle plus x-distance to feature related to the particle
#    mu_init[1,:] = X[1,:] + xy[1,:] # the same for Y
#    return mu_init
    
    #
#def associate_known(Sbar, z, W, Lambda_psi, Q, known_association):
#    #           S_bar(t)            4XM particle set representing the estimated states and weights [x,y,theta, weights]'
#    #           z(t)                2Xn measured range and angle between observed landmark and robot
#    #           W                   2XN coordinates of the landmark in x,y in tth time 
#    #           Lambda_psi          1X1 threshhold parameter for detecting outliers
#    #           Q                   2X2 measurement noise covariance matrix
#    #           known_associations  1Xn association
#    # Outputs: 
#    #           outlier             1Xn boolean = if identified association is an outlier
#    #           Psi(t)              1XnXM most likely value
#    return
#function [outlier,Psi] = associate_known(S_bar,z,W,Lambda_psi,Q,known_associations)
#% dimensions
#n = size(z,2);      % number of observations made n (i)
#N = size(W,2);      % number of landmarks known N (k)
#M = size(S_bar,2);  % number of particles M (m)
#
#% memory allocation
#X = zeros(2,M);
#nu_r = zeros(2,M);
#nu_theta = zeros(2,M);
#nu_theta_mod = zeros(2,M);
#outlier = zeros(1,n);
#Psi_ikm = zeros(N,n,M);
#
#% maximum likelihood data association algorithm
#for i = 1:n 
#    for k=1:N
#        X = observation_model(S_bar,W,k); % predict measurement 2xM
#        nu_r = z(1,i) - X(1,:); % innovation range
#        nu_theta = z(2,i) - X(2,:);  % innovation angle
#        nu_theta_mod = mod(nu_theta+pi,2*pi)-pi; % keep angle error value between -pi and pi
#        Psi_ikm(k,i,:) = det(2*pi*Q)^(-1/2)*exp(-0.5 * ((nu_r).^2/Q(1,1) + (nu_theta_mod).^2 / Q(2,2)));  % likelihood
#    end
#end
#
#% use known associations
#Psi = Psi_ikm(known_associations,:,:);
#
#% maximize the likelihood of the known associations
#if (size(Psi,2) > 1)
#    Psi = max(Psi); % 1xnxM
#end
#
#% detect outliers
#for i =1:n    
#    outlier(i) = mean(Psi(1,i,:)) <= Lambda_psi;
#end
#
#% for debug purposes
#[Psi_check,Psi_index]=max(Psi_ikm);
#outlier_check = Psi_index(:,:,1) ~= known_associations;
#
#% also notice that you have to do something here even if you do not have to maximize the likelihood.
#
#end

