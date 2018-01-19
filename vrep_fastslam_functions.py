
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 00:35:16 2018

@author: Magnus Tarle

Description:
    Contains functions and simplified test functions for vrep_fastslam.py
"""
import numpy as np
import matplotlib.patches as mpatches # used for legend, ellipses and rectangles
from scipy.stats import norm, chi2 # used for the covariance

# USED IN SIM
def cov_ellipse(cov, q=None, nsig=None, **kwargs):
    """
    source: https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
    answered Sep 28 '16 at 13:40 by Syrtis Major

    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations. 
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)

    val, vec = np.linalg.eigh(cov)
    width, height = 2 * np.sqrt(val[:, None] * r2)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    return width, height, rotation

# USED IN SIM
def add_ellipse_object(Sigma, mu):
    # Function adds a matplotlib.matches.ellipse-object based on covariance matrix
    # Inputs:    
    #           Sigma       NXM2X2 Feature covariance matrix (N observed features, M particles) 
    #           mu          2XM observation function for range-bearing measurement, [r, theta]' in robot reference frame
    # Outputs:  
    #           ecov        1XN list of matplotlib.matches.ellipse-objects representing the covariance  
    mu_mean = np.mean(mu,axis=2)
    Sigma_mean = np.mean(Sigma,axis=1)
    ecov = []
    n_features = len(np.mean(Sigma,axis=1))
    for n in range(0,n_features): 
        feat_cov_matrix = Sigma_mean[n]
        featwidth, featheight, featrotdeg = cov_ellipse(feat_cov_matrix, q=0.99)
        featx = mu_mean[n][0]
        featy = mu_mean[n][1]        
        ecov.append(mpatches.Ellipse(xy=(featx, featy), width=featwidth, height=featheight, angle=featrotdeg, fill=0))    
    return ecov

# USED IN SIM
def add_arrow_object(iter,carpos,featpos,feature_index):
    # Function adds a matplotlib.matches.arrow-object between true robot position and feature position    
    # Inputs:    
    #           iter        1X1 simulation iteration index
    #           carpos      1XK list of true car positions
    #           featpos     2X1 observed feature position [x,y]' in world reference frame
    #           feature_index 1X1 assigned index to the feature
    # Outputs:  
    #           a        1XN list of matplotlib.matches.arrows-objects for plotting observed features   
    carpos = np.asarray(carpos)
    feature_index = feature_index -1
    a = mpatches.Arrow(x=carpos[iter,0], y=carpos[iter,1], dx=featpos[feature_index,0]-carpos[iter,0],dy=featpos[feature_index,1]-carpos[iter,1], width = 0.2, color = 'red')
    return a
    
# USED IN SIM
def id_feature(features,sensed_object):
    # Function identifies which feature index is observed given an object handle    
    # Inputs:    
    #           features    1X1 list of features in simulated world
    #           sensed_object 1X1 observed object handle
    # Outputs:  
    #           itemindex   1X1 index of feature object handle in feature list    
    # This function retrieves the index of the observed object handle in given in vrep
    itemindex = np.where(features==sensed_object)
    return int(itemindex[0])+1

# USED IN SIM
def init_mean_from_z(Xbar, z): 
    # This function initializes the mean location mu of the observed feature 
    # in world coordinates based on a range and angle z related to 
    # the particle set X
    # See p.320 in Probabilistic Robotics by Sebastian Thrun
    # The bearing lies in the interval [-pi,pi)
    # Inputs:    
    #           X           3XM previous particle set representing the states and the weights [x,y,theta]
    #           z           2XM observation function for range-bearing measurement, [r, theta]' in robot reference frame
    # Outputs:  
    #           mu_init     2XM position of the feature related to each particle
    M = np.size(Xbar[0,:])     # Number of particles in particle set  
    mu_init = np.zeros((2,M)) 
    mu_init[0,:] = Xbar[0,:] + z[0,:] * np.sin(z[1,:] + Xbar[2,:]) # X position of particle plus x-distance to feature related to the particle
    mu_init[1,:] = Xbar[1,:] + z[0,:] * np.cos(-z[1,:] - Xbar[2,:]) # the same for Y     
    return mu_init

def test_init_mean_from_z():
    # Test function
    M = 1 # Number of particles
    x = 0
    y = 0
    theta = np.pi/4   
    X1 = np.array([x, y, theta]) # Assumed particle start position
    X = np.repeat(X1[:, np.newaxis], M, axis=1) # Set of particles 
    zr = 1
    ztheta = np.pi/2
    Z1 = np.array([zr, ztheta]) # Assumed particle start position
    z = np.repeat(Z1[:, np.newaxis], M, axis=1) # Set of particles 
    mu_init1 = init_mean_from_z(X, z)
    print('car pos x:', X1[0], 'pos y:', X1[1], 'theta:', X1[2], 'deg:', X1[2]*180/np.pi)
    print('observation r:', z[0], 'theta:', z[1], 'deg:', X1[2]*180/np.pi)    
    print('mean x:', mu_init1[0], 'mean y:', mu_init1[1])
# test_init_mean_from_z()
    

# USED IN SIM!    
def observation_model_zhat(X,mu,k,Q): # maybe remove Q and j here....
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
    Featx = mu[k,0,:] # Extract one feature j for creating a distance calculation in x for all particles
    Featy = mu[k,1,:] # Extract one feature j and create a matrix shape for creating a distance calculation in y for all particles
    Xx = X[0,:] # Extract the x position of all particles
    Xy = X[1,:] # Extract the y position of all particles
    Xtheta = X[2,:] # Extract the theta angle of all particles
    ra = np.sqrt((Featx - Xx)**2 +(Featy - Xy)**2) # range to feature for each particle
    ra = ra.reshape(1,M)
    #    theta = np.arctan2(Featy-Xy,Featx-Xx) - Xtheta # angle to observed feature for each particle
    theta = np.arctan2(Featx-Xx,Featy-Xy) - Xtheta # angle to observed feature for each particle    
    theta_lim = ((theta + np.pi) % (2*np.pi)) - np.pi # limit angle between pi and -pi
    theta_lim = theta_lim.reshape(1,M)
    zmeas = np.concatenate((ra, theta_lim), axis = 0)
    # Add diffusion
    rtheta_stddev2 = np.diag(np.sqrt(Q)) # obtain standard deviation square of process noise (1-dimensional array)
    diffusion_normal = np.random.standard_normal((2,M))  # Normal distribution with standard deviation 1 
    diffusion = diffusion_normal * rtheta_stddev2.reshape(2,1) # Normal distribution with standard deviation according to process noise covariance R (reshape sigma to get 3 rows and 1 column for later matrix inner product multiplication)
    z = zmeas + diffusion[:2,:] # estimated states = old states + motion + diffusion    
    z[1,:] = np.mod(z[1,:] +np.pi,2*np.pi) -np.pi
    return z    

def test_observation_model_zhat():
    M = 1 # Number of particles
    x = -0.5
    y = -1
    theta = -np.pi/2
    
    mx = -1.5#1/np.sqrt(2)
    my = -1.0#1/np.sqrt(2) #np.pi/4  
    
    X1 = np.array([x, y, theta]) # Assumed particle start position
    X = np.repeat(X1[:, np.newaxis], M, axis=1) # Set of particles 
    m1 = np.array([mx, my]) # Assumed particle start position
    m2 = np.repeat(m1[:, np.newaxis], M, axis=1) # Set of particles 
    #m = np.repeat(m2[np.newaxis,:,:], 5, axis=0) # Set of particles
    m3 = m2.reshape(1,2,M)
    k = 3 # landmark
    m_init = np.zeros((1,2,M))
    m_init[0,:,:] = m3
    m = np.concatenate((m_init,m_init),axis=0) # NX2XM
    m = np.concatenate((m,m_init),axis=0) # NX2XM
    m = np.concatenate((m,m_init),axis=0) # NX2XM
    m = np.concatenate((m,m_init),axis=0) # NX2XM
    Q = np.eye(2)*0.1
    zhat = observation_model_zhat(X,m,k,Q)
    thetadeg = zhat[1]*180/np.pi
    print('car pos x:', X1[0], 'pos y:', X1[1], 'theta', X1[2])
    print('mean x:', m[k,0], 'y:', m[k,1])    
    print('calc range:', zhat[0], 'theta:', zhat[1] ,', deg:', thetadeg)      
#test_observation_model_zhat()

# USED IN SIM  
def calculate_measurement_jacobian(X,mu,k): 
    # This function calculates the jacobian of the observation model h for a given
    # feature and the particle set. The Jacobian is used
    # later to obtain the measurement covariance matrix Sigma = HxQxH.T  
    # for a given feature and the particle set.
    # The derivative is based on the feature x,y and not the robot itself (robot has x,y and theta)
    # Inputs:
    #           X(t)    3XM estimated states [x,y,theta]'   
    #           mu(t)   NX2XM estimated mean position of features [x,y]'       
    # Outputs:  
    #           H       MX2X2 H is the Jacobian of observation model h in regards to any observation linearized/evaluated at feature position mu_bar
    
    # Extract relevant variables
    M = np.size(X[0,:])     # Number of particles in particle set    
    mux = mu[k,0,:] # Extract one feature j and create a matrix shape for creating a distance calculation in x for all particles
    muy = mu[k,1,:] # Extract one feature j and create a matrix shape for creating a distance calculation in y for all particles
    Xx = X[0,:] # Extract the x position of all particles
    Xy = X[1,:] # Extract the y position of all particles
    
    # calculate square of range distance between robot and feature
    q = (mux - Xx)**2 +(muy - Xy)**2 # 

    # range linearization
    dhr_dmux = (mux-Xx) / np.sqrt(q)
    dhr_dmuy = (muy-Xy) / np.sqrt(q)
    
    # angle linearization
    dhtheta_dmux = -(mux-Xx) / q #1XM
    dhtheta_dmuy = (muy-Xy) / q
    
    # Reshape matrices to create Jacobian
    dhr_dmux2 = dhr_dmux.reshape(M,1,1)
    dhr_dmuy2 = dhr_dmuy.reshape(M,1,1)  
    dhtheta_dmux2 = dhtheta_dmux.reshape(M,1,1)
    dhtheta_dmuy2 = dhtheta_dmuy.reshape(M,1,1)

    # Organize Jacobian
    H1 = np.concatenate((dhr_dmux2,dhr_dmuy2),axis=2)
    H2 = np.concatenate((dhtheta_dmux2,dhtheta_dmuy2),axis=2)
    H = np.concatenate((H1,H2), axis=1) # MX2X2
    return H

# USED IN SIM
def z_from_detectection(X,observed_objects_pos): 
    # This function calculates a range r and angle theta given x, y in robot reference frame
    # Note: The bearing theta lies in the interval [-pi,pi) in relation to the particle set X
    # Inputs:
    #           X           3XM previous particle set representing the states [x,y,theta]'
    #           observed_objects_pos  2X1 coordinates of the features in x,y in tth time  
    # Outputs:  
    #           zmeas       2XM range r and angle between robot and feature object, [r, theta]'
    M = np.size(X[0,:])     # Number of particles in particle set    
    Featx = np.ones((1,M)) * observed_objects_pos[0,0] # Extract one feature j and create a matrix shape for creating a distance calculation in x for all particles
    Featy = np.ones((1,M)) * observed_objects_pos[1,0] # Extract one feature j and create a matrix shape for creating a distance calculation in y for all particles
    r = np.sqrt((Featx)**2 +(Featy)**2) # range to feature for each particle
    theta = np.arctan2(Featx,Featy) # angle to observed feature for each particle
    theta_lim = ((theta + np.pi) % (2*np.pi)) - np.pi # limit angle between pi and -pi
    zmeas = np.concatenate((r, theta_lim), axis = 0) 
    return zmeas

def test_z_from_detection():
    M = 1 # Number of particles
    x = 0
    y = 0
    theta = 0#2*np.pi
    X1 = np.array([x, y, theta]) # Assumed particle start position
    X = np.repeat(X1[:, np.newaxis], M, axis=1) # Set of particles 
    ox = 0#+1/np.sqrt(2)
    oy = 1 # +1/np.sqrt(2) #np.pi/4
    o1 = np.array([ox, oy]) # Assumed particle start position
    o = np.repeat(o1[:, np.newaxis], M, axis=1) # Set of particles 
    z = z_from_detectection(X, o)
    thetadeg = z[1]*180/np.pi
    print('car pos x:', X1[0], 'pos y:', X1[0], 'theta', X1[2])
    print('observation x:', o[0], 'y:', o[1])    
    print('calc range:', z[0], 'theta:', z[1] ,', deg:', thetadeg)      
#test_z_from_detection()

# USED IN SIM
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
    
    # Extract relevant variables
    M = np.size(X[0,:])             # Number of particles in particle set
    xytheta_dim = 3 # Number of states of each particle 
    Xbar = np.zeros((3,M))
    theta_prev = X[2,:]  # theta angle at previous time step

    # Estimate next x,y position
    xy_predmotion = delta_t * np.array([ v * np.sin(theta_prev), v * np.cos(theta_prev) ]) # motion model on x and y for each particle  
    
    # Estimate orientation
    theta_predmotion = delta_t * w * np.ones((1,M))  #motion model for the angle of each particle 
    
    # Re-organize matrix
    xytheta_predmotion = np.concatenate((xy_predmotion, theta_predmotion), axis = 0) # combine the complete motion model x,y and theta into one matrix 
    
    # Add noise to prediction
    xytheta_stddev2 = np.diag(np.sqrt(R)) # obtain standard deviation of process noise (1-dimensional array)
    diffusion_normal = np.random.standard_normal((xytheta_dim,M))  # Normal distribution with standard deviation 
    diffusion = diffusion_normal * xytheta_stddev2.reshape(3,1) # Normal distribution with standard deviation according to process noise covariance R (reshape sigma to get 3 rows and 1 column for later matrix inner product multiplication)
    Xbar = X + xytheta_predmotion + diffusion # estimated states = old states + motion + diffusion
    
    # Correct angle to be within -pi and pi
    Xbar[2,:] = ((Xbar[2,:] + np.pi) % (2*np.pi)) - np.pi
    return Xbar  
    
# USED IN SIM
def calculate_odometry(delta_angle_R, delta_angle_L, B, R_R, R_L, delta_t, CALIB_ODOM):
    # This function calculates translational speed and angular frequency w of the robot provided
    # from the odometry information of the robot wheel motors
    # Inputs:
    #       delta_angle_R(t): 1X1 angle difference for the left wheel in the tth time step
    #       delta_angle_L(t): 1X1 angle difference for the right wheel in the tth time step
    #       B:                1X1 wheel base (distance between the contact points of the wheels, front and back)
    #       R_L:              1X1 radius of the left wheels
    #       R_R:              1X1 radius of the right wheels
    #       delta_t:          1X1 previous state in [x, y, theta]'
    #       CALIB_ODOM:       1X1 calibration constant for odometry slip etc
    # Outputs:
    #      v(t):             1X1 translational velocity of the robot in the tth time step
    #      w(t):             1X1 angular velocity of the robot in the tth time step
    w_R = delta_angle_R / delta_t
    w_L = delta_angle_L / delta_t
    w = -(w_R*R_R - w_L*R_L) / B * CALIB_ODOM
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
    
# USED IN SIM
def predict_motion_xytheta(x, y, theta, v, w, delta_t):
    # This function performs a prediction step without diffusion, i.e.
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

    x_predmotion = x + v*delta_t*np.sin(theta)
    y_predmotion = y + v*delta_t*np.cos(theta)
    theta_predmotion = (((theta + w*delta_t) + np.pi) % (2*np.pi)) - np.pi # 
    return x_predmotion, y_predmotion, theta_predmotion
    

