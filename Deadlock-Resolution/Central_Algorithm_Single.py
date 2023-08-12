#!/usr/bin/env python3

import rospy
import math
import time
import rosnode
# Import message file
from std_msgs.msg import String
from khepera_communicator.msg import K4_controls, SensorReadings, Opt_Position
from geometry_msgs.msg import PoseStamped

import tf_conversions
import tf2_ros
import numpy as np
#from qpsolvers import solve_qp
#import rps.robotarium as robotarium

from utilities.controllers import *
from utilities.barrier_certificates import *
from utilities.misc import *
from utilities.transformations import *
start = time.time()

rospy.init_node('Algorithm_PrSBC_Central', anonymous=True)

# Get all the node names for all the currently running K4_Send_Cmd nodes (all running Kheperas)
# Get the node names of all the current running nodes
node_list = rosnode.get_node_names()

# Find the nodes that contains the "K4_Send_Cmd_" title
khep_node_list = [s for s in node_list if "K4_Send_Cmd_" in s]


#ip_num_list = [x[13:16] for x in khep_node_list]
ip_num_list = [x[13:15] if x[15] == '_' else x[13:16] for x in khep_node_list]


khep_node_cnt = len(khep_node_list)

# Establish all the publishers to each "K4_controls_" topic, corresponding to each K4_Send_Cmd node, which corresponds to each Khepera robot
pub = {}
for i in range(khep_node_cnt):
	pub.update({str(ip_num_list[i]) : rospy.Publisher('K4_controls_' + str(ip_num_list[i]), K4_controls, queue_size = 10)})
	
# Instantiate Robotarium object
N = 4
aa = np.pi

atIniPos = 0
initial_conditions = np.array([[0., 0., -1., 1.], [1., -1., 0., 0.], [-math.pi / 2, math.pi / 2, 0., math.pi]])
goal_points = np.array([[0., 0., 1., -1.], [-1., 1., 0., 0.], [math.pi / 2, -math.pi / 2, math.pi, 0.]])

dxu = np.array([[0,0,0,0],[0,0,0,0]])

Kp = 10 # TODO: might need to scale output depending on how large output of QP is 
Vmax = 200
Wmax = np.pi
iflag = np.zeros(khep_node_cnt)
flag = np.zeros(khep_node_cnt) # stop flag


# PrSBC Parameters
safety_radius = 0.2
barrier_gain = 1
projection_distance = 0.05
magnitude_limit = 0.2
x = np.array([[0,0,0,0], [0,0,0,0], [0,0,0,0]])
dxu = np.zeros((2, N))
Omega = math.pi/10*np.ones((N))


# Create single integrator position controller
single_integrator_position_controller = create_si_position_controller()

si_barrier_cert = de_create_single_integrator_CLF_CBF_CBF2(safety_radius=1.7/2.)
si_barrier_cert2 = create_single_integrator_barrier_certificate(safety_radius=0.17)
_, uni_to_si_states = create_si_to_uni_mapping()

si_to_uni_dyn = create_si_to_uni_dynamics()



uni_controller = create_clf_unicycle_pose_controller()
uni_barrier_cert = create_unicycle_barrier_certificate()


def control_robots(x0, x1, x2, x3, y0, y1, y2, y3, t, t0, t1, t2, t3, goals,p):
	global dxu
	global atIniPos
	global iflag
	N =4
	# reshape poses so that its 3 x n
	x = np.array([[x0, x1, x2, x3], [y0, y1, y2, y3], [t0, t1, t2, t3]])
	
	if atIniPos == 0:
		di = np.sqrt((initial_conditions[0][:4] - x[0][:4]) ** 2 + (initial_conditions[1][:4] - x[1][:4]) ** 2)
		
		print("checking if ini pos true")
		
		if (di < .075).all() or (iflag == 1).all():

			V = 0
			W = 0
			atIniPos = 1
			print("ini pos true")
		
			return V, W
		else:
			dxu = uni_controller(x, initial_conditions)
			dxu = uni_barrier_cert(dxu,x)
			for i in range(khep_node_cnt):
				if round(di[i], 2) < .05 or iflag[i] == 1:
		    			dxu[0, i] = 0
		    			dxu[1, i] = 0
		    			iflag[i] = 1
			return dxu[0]*700, dxu[1]
		
	
#################################	
	if atIniPos == 1:
		print("atinipos has met")
		# get distance to gaol for all robots
		d = np.sqrt((goals[0][:4] - x[0][:4]) ** 2 + (goals[1][:4] - x[1][:4]) ** 2)

		# stop if distance threshold is met

		if (d < .075).all() or (flag == 1).all():

			V = 0
			W = 0
			sub.unregister()
				
			return V, W
		x_si = uni_to_si_states(x)
		# robot i
		xx = np.reshape(x[:, p], (3, 1))
		xi = np.reshape(x_si[:, p], (2, 1))
		mask = np.arange(x_si.shape[1]) != p
		xo = x_si[:, mask]  # for obstacles
		xgoal = goal_points[0:2, p].reshape((2,-1))
		dxx = si_barrier_cert(xi*5, xo*5, xgoal*5, Omega[p])
		if dxx is None:
		    dxx = [0,0,0,math.pi/2]
		    print(p)
		dx = np.array([[dxx[0]],[dxx[1]]])
		Omega[p] = dxx[3]
		du = si_to_uni_dyn(dx, xx)
			
		dxu[0, p] = du[0, 0]
		dxu[1, p] = du[1, 0]
		for i in range(khep_node_cnt):
        		if round(d[i], 2) < .05 or flag[i] == 1:
            			dxu[0, i] = 0
            			dxu[1, i] = 0
            			flag[i] = 1

		return dxu[0, p] * 200, dxu[1, p]/10.

# This callback function is where the centralized swarm algorithm, or any algorithm should be
# data is the info subscribed from the vicon node, contains the global position, velocity, etc
# the algorithm placed inside this callback should be published to the K4_controls topics
# which should have the K4_controls message type:
# Angular velocity: ctrl_W
# Linear velocity: ctrl_V
def callback(data):
	global atIniPos
	# arrange positions and goals into PrSBC form
	
	# call control_robots with arranged positions/goals and return V and W list
	if atIniPos == 0:
		p = 0
		V, W = control_robots(data.x[0],data.x[1],data.x[2],data.x[3], data.y[0],data.y[1],data.y[2],data.y[3], data.theta ,data.theta[0],data.theta[1],data.theta[2],data.theta[3], goal_points,p)
		
		for i in range(khep_node_cnt):
			# create control message
			control_msgs = K4_controls()
			
			# set appropriate velocities
			control_msgs.ctrl_V = V[i]
			control_msgs.ctrl_W = W[i]

			# send data
			print('-------------------------')
			print('Control for robot %s' % data.ip[i])
			print(control_msgs)
			#print(atIniPos,"atIniPos")
			pub.get(str(data.ip[i])).publish(control_msgs)
		
	if atIniPos == 1:
		for p in range(4):
			V, W = control_robots(data.x[0],data.x[1],data.x[2],data.x[3], data.y[0],data.y[1],data.y[2],data.y[3], data.theta ,data.theta[0],data.theta[1],data.theta[2],data.theta[3], goal_points, p)


			# send V and W list

			# create control message
			control_msgs = K4_controls()
			# set appropriate velocities
			control_msgs.ctrl_V = V
			control_msgs.ctrl_W = W

			# send data
			print('-------------------------')
			print('Control for robot %s' % data.ip[p])
			print(control_msgs)
			pub.get(str(data.ip[p])).publish(control_msgs)


def central():
	global sub

	# Set up the Subscribers
	print('Set up the Subscribers')
	sub = rospy.Subscriber('/K4_Mailbox', Opt_Position, callback)
	
	print('Entering Waiting loop')
	while not rospy.is_shutdown():
		rospy.sleep(1)
		if ((flag == 1).all()):
			print('Program Complete! Please press ^C')


if __name__ == '__main__':
	try:
		central()
	except rospy.ROSInterruptException:
		print(rospy.ROSInterruptException)
