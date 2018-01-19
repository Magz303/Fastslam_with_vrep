#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 17:35:29 2017

@author: Magnus Tarle

Description:
    This file contains the robot control which makes the robot in Vrep move by
    setting the angular velocity of the front wheels
    The thread class is based on an internet source which I couldn't 
    find the reference to
"""

import vrep
import threading
import time

exitFlag = 0

# This class might be remade to be an I/O class to vrep, might have to refactor here later
class ctrlRobot (threading.Thread):
   def __init__(self, clientID, h_motor_left, h_motor_right, name):
      threading.Thread.__init__(self)
      self.clientID = clientID
      self.h_motor_left = h_motor_left
      self.h_motor_right = h_motor_right
      self.name = name
   def run(self):
      print("Starting " + self.name)
      run_path(self.clientID, self.h_motor_left, self.h_motor_right)
      print("Exiting " + self.name)
   
def run_path(clientID, h_motor_left, h_motor_right):
    # Main path for robot
    vel0 = 3
    for m in range(0, 8):
        # run straight
        vel_left = vel0
        vel_right = vel0
        error_code2 = vrep.simxSetJointTargetVelocity(clientID,h_motor_left,vel_left,vrep.simx_opmode_streaming)       
        error_code2 = vrep.simxSetJointTargetVelocity(clientID,h_motor_right,vel_right,vrep.simx_opmode_streaming)
        time.sleep(4)
        # make a turn
        vel_left = vel0 + vel0
        vel_right = vel0 - vel0 + 0.01
        error_code2 = vrep.simxSetJointTargetVelocity(clientID,h_motor_left,vel_left,vrep.simx_opmode_streaming)       
        error_code2 = vrep.simxSetJointTargetVelocity(clientID,h_motor_right,vel_right,vrep.simx_opmode_streaming)
        time.sleep(1.65)
        m = m + 1
        
def run_path2(clientID, h_motor_left, h_motor_right):
    # Alternative path for robot
    vel0 = 3
    # run straight
    vel_left = vel0
    vel_right = 0
    error_code2 = vrep.simxSetJointTargetVelocity(clientID,h_motor_left,vel_left,vrep.simx_opmode_streaming)       
    error_code2 = vrep.simxSetJointTargetVelocity(clientID,h_motor_right,vel_right,vrep.simx_opmode_streaming)
    time.sleep(2.6)
    # make a turn
    vel_left = 0
    vel_right = 0
    error_code2 = vrep.simxSetJointTargetVelocity(clientID,h_motor_left,vel_left,vrep.simx_opmode_streaming)       
    error_code2 = vrep.simxSetJointTargetVelocity(clientID,h_motor_right,vel_right,vrep.simx_opmode_streaming)
    time.sleep(1.65)        
