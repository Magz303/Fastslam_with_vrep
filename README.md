# Fastslam_with_vrep
Description:
    Course project which uses the Fastslam 1.0 algorithm with known 
    correspondences together with a robot car model in the 
    robot simulation software V-rep. 

Requirements: 
    *Vrep software, specific Vrep model file, vrep api library
    *vrep_fastslam_functions.py for related functions used throughout the code
    *robotctrl.py to control the robot path
    
Usage:
    Start the simulation in Vrep and execute the code. It will run for
    a given simulation time. The user has to decide on amount on particles
    and noise and process covariance matrix.

Note1: 
    The plotting assumes that a separate plot window will be used and 
    not an inline plot

Note2: 
    The code is procedural and not object oriented. It follows the 
    algoritm in "Probabilistic Robotics" by Sebastian Thrun et.al.. I made an
    attempt to avoid iteration across the particles to save speed.

Note3: 
    The code needs refactoring to be more clear
