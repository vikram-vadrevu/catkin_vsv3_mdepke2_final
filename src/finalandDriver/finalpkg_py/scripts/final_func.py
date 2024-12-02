#!/usr/bin/env python
import numpy as np
from scipy.linalg import expm, logm
from final_header import *
from ModernRobotics import *

"""
Use 'expm' for matrix exponential.
Angles are in radian, distance are in meters.
"""
def Get_MS():
   # =================== Your code starts here ====================#
   # Fill in the correct values for S1~6, as well as the M matrix


   P_BS = np.array([(540-150)/1000, (120-93+83+82+59+150)/1000, (10+152+53.5)/1000])
   M  = np.array([[0,-1, 0, P_BS[0]],
                  [0, 0,-1, P_BS[1]],
                  [1, 0, 0, P_BS[2]],
                  [0, 0, 0, 1]])
  
   S = np.array([[0,0,1,0.15,0.15,0],
                 [0,1,0,-0.162,0,-0.15],
                 [0,1,0,-0.162,0,0.094],
                 [0,1,0,-0.162,0,0.307],
                 [1,0,0,0,0.162,-0.26],
                 [0,1,0,-0.162,0,0.39]]).T

   return M, S

"""
Function that calculates encoder numbers for each motor
"""
def lab_fk(theta1, theta2, theta3, theta4, theta5, theta6):
   """
   Function that calculates encoder numbers for each motor
   """
   # Initialize the return_value
   return_value = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
   theta = return_value
   # =========== Implement joint angle to encoder expressions here ===========
   print("Foward kinematics calculated:\n")

   M, S = Get_MS()
   T = FKinSpace(M, S, theta)


   print(str(T) + "\n")


   return_value[0] = theta1 + PI
   return_value[1] = theta2
   return_value[2] = theta3
   return_value[3] = theta4 - (0.5*PI)
   return_value[4] = theta5
   return_value[5] = theta6


   return return_value




	# ==============================================================#


"""
Function that calculates an elbow up Inverse Kinematic solution for the UR3
"""
def lab_invk(xWgrip, yWgrip, zWgrip, yaw_WgripDegree):
	# =================== Your code starts here ====================#
	xWgrip = xWgrip + .150
	yWgrip = yWgrip - .150
	zWgrip = zWgrip - .010
	links = np.array([-1000, 152, 120, 244, 93, 213, 83, 83, 82, 53.5, 59])/1000
	# print(f'Links: {links}')
	x_center = xWgrip - links[9]*np.cos(np.radians(yaw_WgripDegree))
	y_center = yWgrip - links[9]*np.sin(np.radians(yaw_WgripDegree))
	z_center = zWgrip

	theta1 = np.arctan2(y_center, x_center) - np.arcsin((0.110) / np.sqrt(x_center**2 + y_center**2))
	theta6 = (theta1 + np.pi/2) - np.radians(yaw_WgripDegree)

	rotation = np.array([[-np.cos(theta1),  np.sin(theta1)],
					  	 [-np.sin(theta1), -np.cos(theta1)]])
	
	# x_3end = x_center + links[7] * -1*np.cos(theta1) + (links[6] - 0.027)*np.sin(theta1))
	# y_3end = y_center + links[7] *  -1*np.sin(theta1) + (links[6] - 0.027)*-1*np.cos(theta1))

	offset_dist = np.array([[links[7]],
						    [links[6] + 0.027]])
	
	pos_3end = np.array([[x_center],[y_center]]) + (rotation @ offset_dist)
	
	x_3end, y_3end = pos_3end.reshape(2,).tolist()
	z_3end = z_center + links[10] + links[8]

	c_line = np.sqrt(x_3end**2 + y_3end**2 + (z_3end - links[1])**2)

	theta3 = (np.pi) - (np.arccos((links[5]**2 + links[3]**2 - c_line**2)/(2*links[3]*links[5])))

	theta2small = np.arccos(np.sqrt(x_3end**2 + y_3end**2)/c_line)

	theta2big = np.arccos((links[3]**2 + c_line**2 - links[5]**2) / (2*links[3]*c_line))
	
	theta2 = -theta2small - theta2big
	theta4 = -theta2 - theta3
	theta5 = -np.pi/2


	# ==============================================================#
	return lab_fk(theta1, theta2, theta3, theta4, theta5, theta6)