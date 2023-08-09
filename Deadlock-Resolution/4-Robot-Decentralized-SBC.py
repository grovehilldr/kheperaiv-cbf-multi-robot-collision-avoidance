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
from qpsolvers import solve_qp

# from rps.utilities.controllers import *
from utilities.pr_barrier_certs import *


from rps.utilities.controllers import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.transformations import *
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

# Robot dynamics variables
uni_goals = np.array([[1.0,1.0,-1.0,-1.0  ], [1.0,-1.0,-1.0,1.0], [0.1,0.1,0.1,0.1]]) # changes based on Optitrack setup
Kp = 10 # TODO: might need to scale output depending on how large output of QP is 
Vmax = 200
Wmax = np.pi
flag = np.zeros(khep_node_cnt) # stop flag

# PrSBC Parameters
safety_radius = 0.2
barrier_gain = 1
projection_distance = 0.05
magnitude_limit = 0.2


single_integrator_position_controller = create_si_position_controller()

si_barrier_cert = de_create_single_integrator_barrier_certificate()

_, uni_to_si_states = create_si_to_uni_mapping()

si_to_uni_dyn = create_si_to_uni_dynamics()
'''
v_rand_span = 0.005 * np.ones((2, n)) # setting up velocity error range for each robot
x_rand_span_x = 0.02 * np.random.randint(3, 4, (1, n)) # setting up position error range for each robot,
x_rand_span_y = 0.02 * np.random.randint(1, 4, (1, n)) # rand_span serves as the upper bound of uncertainty for each of the robot
x_rand_span_xy = np.concatenate((x_rand_span_x, x_rand_span_y))
'''

def control_robots(x0, x1, x2, x3, y0, y1, y2, y3, t0, t1, t2, t3, goals,p):
	N =4
	# reshape poses so that its 3 x n
	x = np.array([[x0, x1, x2, x3], [y0, y1, y2, y3], [t0, t1, t2, t3]])

	# TODO: Add noise

	# get distance to gaol for all robots
	d = np.sqrt((goals[0][:4] - x[0][:4]) ** 2 + (goals[1][:4] - x[1][:4]) ** 2)

	# stop if distance threshold is met

	if (d < .075).all() or (flag == 1).all():

		V = 0
		W = 0
		sub.unregister()
		return V, W


	x = np.array([[x0, x1, x2, x3], [y0, y1, y2, y3], [t0, t1, t2, t3]])
	x_si = uni_to_si_states(x)

	# Create single-integrator control inputs
	dxi = single_integrator_position_controller(x_si, uni_goals[:2][:])

	# robot i
	xx = np.reshape(x[:, p], (3, 1))
	xi = np.reshape(x_si[:, p], (2, 1))
	mask = np.arange(x_si.shape[1]) != p
	xo = x_si[:, mask]  # for obstacles
	dx = dxi[:, p]
	dx = si_barrier_cert(dx, xi, xo)

	du = si_to_uni_dyn(dx, xx)
	dxu = np.zeros((2, N))
	dxu[0, p] = du[0, 0]
	dxu[1, p] = du[1, 0]


	return dxu[0, p] * 500, dxu[1, p]/10.

# This callback function is where the centralized swarm algorithm, or any algorithm should be
# data is the info subscribed from the vicon node, contains the global position, velocity, etc
# the algorithm placed inside this callback should be published to the K4_controls topics
# which should have the K4_controls message type:
# Angular velocity: ctrl_W
# Linear velocity: ctrl_V
def callback(data):
	# arrange positions and goals into PrSBC form
	
	# call control_robots with arranged positions/goals and return V and W list
	for p in range(4):
		V, W = control_robots(data.x[0],data.x[1],data.x[2],data.x[3], data.y[0],data.y[1],data.y[2],data.y[3], data.theta[0],data.theta[1],data.theta[2],data.theta[3], uni_goals, p)


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
