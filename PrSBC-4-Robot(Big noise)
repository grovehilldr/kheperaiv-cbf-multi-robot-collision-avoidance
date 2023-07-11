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
from utilities.controllers import *
from utilities.util import filter_ip

def trap_cdf_inv(a, c, delta, sigma):
    # returns list of b2, b1, sigma
    b2 = delta
    b1 = delta

    # a and c should be positive

    if a > c: # [-A, A] is the large one, and[-C, C] is the smaller one
        A = a
        C = c
    else:
        A = c
        C = a

    if A == 0 and C == 0:
        return b2, b1, sigma

    # O_vec = [-(A + C), -(A - C), (A - C), (A + C)] # vector of vertices on the trap distribution cdf

    h = 1 / (2 * A) # height of the trap distribution
    area_seq = [1/2 * 2 * C * h, 2 * (A - C) * h, 1/2 * 2 * C * h]
    area_vec = [area_seq[0], sum(area_seq[:2])]

    if abs(A - C) < 1e-5: # then is triangle
        # assuming sigma > 50
        b1 = (A + C) - 2 * C * np.sqrt((1 - sigma) / (1 - area_vec[1])) # 1 - area_vec[1] should be very close to 0.5
        b2 = -b1

        b1 = b1 + delta
        b2 = b2 + delta # apply shift here due to xi - xj

    else: # than is trap
        if sigma > area_vec[1]: # right triangle area
            b1 = (A + C) - 2 * C * np.sqrt((1 - sigma) / (1 - area_vec[1]))
            b2 = -(A + C) + 2 * C * np.sqrt((1 - sigma) / (1 - area_vec[1]))

            b1 = b1 + delta
            b2 = b2 + delta # apply shift here due to xi - xj

        elif sigma > area_vec[0] and sigma <= area_vec[1]: # in between the triangle part
            b1 = -(A - C) + (sigma - area_vec[0]) / h # assuming > 50%, then b1 should > 0
            b2 = -b1

            b1 = b1 + delta
            b2 = b2 + delta # apply shift here due to xi - xj

            # note that b1 could be > or < b2, depending on whether sigma > or < .5

        elif sigma <= area_vec[0]:
            b1 = -(A + C) + 2 * C * np.sqrt(sigma / area_vec[0]) # assuming > 50%, then b1 should > 0
            b2 = -b1

            b1 = b1 + delta
            b2 = b2 + delta # apply shift here due to xi - xj

        else:
            print('first triangle, which is not allowed as long as we assume sigma > 50%')

    return b2, b1, sigma
dxi = np.array([[0,0,0,0],[0,0,0,0]])

x = np.array([[0,0,0,0],[0,0,0,0]])
N = dxi.shape[1]
v_rand_span = 0.005 * np.ones((2, N)) # setting up velocity error range for each robot


x_rand_span_x = 0.08 * np.random.randint(3, 4, (1, N)) # setting up position error range for each robot,
x_rand_span_y = 0.08 * np.random.randint(1, 4, (1, N)) # rand_span serves as the upper bound of uncertainty for each of the robot

x_rand_span_xy = np.concatenate((x_rand_span_x, x_rand_span_y))

def create_si_pr_barrier_certificate_centralized(gamma = 100, safety_radius = 0.2, magnitude_limit = 0.2,Confidence = 0.8,XRandSpan=None, URandSpan=None):
    if URandSpan is None:
        URandSpan = [0]
    if XRandSpan is None:
        XRandSpan = [0]
    def barrier_certificate(dxi, x,XRandSpan, URandSpan):
 

        
        #N = dxi.shape[1]
        num_constraints = int(comb(N, 2))
        A = np.zeros((num_constraints, 2 * N))
        b = np.zeros(num_constraints)
        H = sparse(matrix(2 * np.identity(2 * N)))

        count = 0
        if len(XRandSpan) == 1:
            XRandSpan = np.zeros(2, N)
        if len(URandSpan) == 1:
            URandSpan = np.zeros(2, N)
        for i in range(N - 1):
            for j in range(i + 1, N):

                max_dvij_x = np.linalg.norm(URandSpan[0, i] + URandSpan[0, j])
                max_dvij_y = np.linalg.norm(URandSpan[1, i] + URandSpan[1, j])
                max_dxij_x = np.linalg.norm(x[0, i] - x[0, j]) + np.linalg.norm(XRandSpan[0, i] + XRandSpan[0, j])
                max_dxij_y = np.linalg.norm(x[1, i] - x[1, j]) + np.linalg.norm(XRandSpan[1, i] + XRandSpan[1, j])
                BB_x = -safety_radius ** 2 - 2 / gamma * max_dvij_x * max_dxij_x
                BB_y = -safety_radius ** 2 - 2 / gamma * max_dvij_y * max_dxij_y
                b2_x, b1_x, sigma = trap_cdf_inv(XRandSpan[0, i], XRandSpan[0, j], x[0, i] - x[0, j], Confidence)
                b2_y, b1_y, sigma = trap_cdf_inv(XRandSpan[1, i], XRandSpan[1, j], x[1, i] - x[1, j], Confidence)

                if (b2_x < 0 and b1_x > 0) or (b2_x > 0 and b1_x < 0):
                    # print('WARNING: distance between robots on x smaller than error bound!')
                    b_x = 0
                elif (b1_x < 0) and (b2_x < b1_x) or (b2_x < 0 and b2_x > b1_x):
                    b_x = b1_x
                elif (b2_x > 0 and b2_x < b1_x) or (b1_x > 0 and b2_x > b1_x):
                    b_x = b2_x
                else:
                    b_x = b1_x
                    # print('WARNING: no uncertainty or sigma = 0.5 on x')  # b1 = b2 or no uncertainty

                if (b2_y < 0 and b1_y > 0) or (b2_y > 0 and b1_y < 0):
                    # print('WARNING: distance between robots on y smaller than error bound!')
                    b_y = 0
                elif (b1_y < 0 and b2_y < b1_y) or (b2_y < 0 and b2_y > b1_y):
                    b_y = b1_y
                elif (b2_y > 0 and b2_y < b1_y) or (b1_y > 0 and b2_y > b1_y):
                    b_y = b2_y
                else:
                    b_y = b1_y

                A[count, (2 * i)] = -2 * b_x  # matlab original: A(count, (2*i-1):(2*i)) = -2*([b_x;b_y]);
                A[count, (2 * i + 1)] = -2 * b_y

                A[count, (2 * j)] = 2 * b_x  # matlab original: A(count, (2*j-1):(2*j)) =  2*([b_x;b_y])';
                A[count, (2 * j + 1)] = 2 * b_y

                h1 = np.linalg.norm([b_x, 0.0]) ** 2 - safety_radius ** 2 - 2 * np.linalg.norm(
                    [max_dvij_x, 0]) * np.linalg.norm([max_dxij_x, 0]) / gamma
                h2 = np.linalg.norm([0, b_y]) ** 2 - safety_radius ** 2 - 2 * np.linalg.norm(
                    [0, max_dvij_y]) * np.linalg.norm([0, max_dxij_y]) / gamma  # h_y

                h = h1 + h2

                b[count] = gamma * h ** 3  # matlab original: b(count) = gamma*h^3
                count += 1

        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, 0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] =dxi[:, idxs_to_normalize] * (magnitude_limit / norms[idxs_to_normalize])

        f_mat = -2 * np.reshape(dxi, 2 * N, order='F')
        f_mat = f_mat.astype('float')
        result = qp(H, matrix(f_mat), matrix(A), matrix(b))['x']

        return np.reshape(result, (2, -1), order='F')
    return barrier_certificate
    
def create_pr_unicycle_barrier_certificate_cent(barrier_gain=100, safety_radius=0.12, projection_distance=0.05, magnitude_limit=0.2, Confidence = 0.8,XRandSpan=None, URandSpan=None):
    """
    MODIFIED VERSION OF create_unicycle_barrier_certificate FROM ROBOTARIUM

    Creates a unicycle Probability Safety barrier cetifcate to avoid collisions. Uses the diffeomorphism mapping
    and single integrator implementation. For optimization purposes, this function returns
    another function.

    barrier_gain: double (how fast the robots can approach each other)
    safety_radius: double (how far apart the robots should stay)
    projection_distance: double (how far ahead to place the bubble)

    -> function (the unicycle barrier certificate function)

    CREATED BY: Robert Wilk
    LAST MODIFIED: 10/19/2022
    """

    #Check user input types
    assert isinstance(barrier_gain, (int, float)), "In the function create_pr_unicycle_barrier_certificate, the barrier gain (barrier_gain) must be an integer or float. Recieved type %r." % type(barrier_gain).__name__
    assert isinstance(safety_radius, (int, float)), "In the function create_pr_unicycle_barrier_certificate, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(safety_radius).__name__
    assert isinstance(projection_distance, (int, float)), "In the function create_pr_unicycle_barrier_certificate, the projected point distance for the diffeomorphism between sinlge integrator and unicycle (projection_distance) must be an integer or float. Recieved type %r." % type(projection_distance).__name__
    assert isinstance(magnitude_limit, (int, float)), "In the function create_pr_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(magnitude_limit).__name__
    assert isinstance(confidence_level, float), "In the function create_pr_unicycle_barrier_certificate, the confidence level must be a float. Recieved type %r." % type(confidence_level).__name__

    #Check user input ranges/sizes
    
    assert barrier_gain > 0, "In the function create_pr_unicycle_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert safety_radius >= 0.12, "In the function create_pr_unicycle_barrier_certificate, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m). Recieved %r." % safety_radius
    assert projection_distance > 0, "In the function create_pr_unicycle_barrier_certificate, the projected point distance for the diffeomorphism between sinlge integrator and unicycle (projection_distance) must be positive. Recieved %r." % projection_distance
    assert magnitude_limit > 0, "In the function create_pr_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_pr_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit
    assert confidence_level <= 1, "In the function create_pr_unicycle_barrier_certificate, the confidence level must be less than 1. Recieved %r." % confidence_level
    assert confidence_level >= 0, "In the function create_pr_unicycle_barrier_certificate, the confidence level must be positive (greater than 0). Recieved %r." % confidence_level

    si_barrier_cert = create_si_pr_barrier_certificate_centralized(gamma=barrier_gain, safety_radius=safety_radius+projection_distance, Confidence=confidence_level,XRandSpan=XRandSpan, URandSpan=URandSpan)

    si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping(projection_distance=projection_distance)

    uni_to_si_dyn = create_uni_to_si_dynamics(projection_distance=projection_distance)

    def f(dxu, x, XRandSpan=None, URandSpan=None):

        if URandSpan is None:
            URandSpan = np.zeros((2, x.shape[1]))
        if XRandSpan is None:
            XRandSpan = np.zeros((2, x.shape[1]))

        #Check user input types
        assert isinstance(dxu, np.ndarray), "In the function created by the create_unicycle_barrier_certificate function, the unicycle robot velocity command (dxu) must be a numpy array. Recieved type %r." % type(dxu).__name__
        assert isinstance(x, np.ndarray), "In the function created by the create_unicycle_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

        #Check user input ranges/sizes
        assert x.shape[0] == 3, "In the function created by the create_unicycle_barrier_certificate function, the dimension of the unicycle robot states (x) must be 3 ([x;y;theta]). Recieved dimension %r." % x.shape[0]
        assert dxu.shape[0] == 2, "In the function created by the create_unicycle_barrier_certificate function, the dimension of the robot unicycle velocity command (dxu) must be 2 ([v;w]). Recieved dimension %r." % dxu.shape[0]
        assert x.shape[1] == dxu.shape[1], "In the function created by the create_unicycle_barrier_certificate function, the number of robot states (x) must be equal to the number of robot unicycle velocity commands (dxu). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxu.shape[0], dxu.shape[1])


        x_si = uni_to_si_states(x)
        #Convert unicycle control command to single integrator one
        dxi = uni_to_si_dyn(dxu, x)
        #Apply single integrator barrier certificate
        dxi = si_barrier_cert(dxi, x_si, XRandSpan, URandSpan)
        #Return safe unicycle command
        return si_to_uni_dyn(dxi, x)

    return f
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
uni_goals = np.array([[1,1,-1,-1  ], [1,-1,-1,1], [0,0,0,0]]) # changes based on Optitrack setup
Kp = 10 # TODO: might need to scale output depending on how large output of QP is 
Vmax = 200
Wmax = np.pi
flag = np.zeros(khep_node_cnt) # stop flag

# PrSBC Parameters
safety_radius = 0.13
barrier_gain = 1000
projection_distance = 0.05
magnitude_limit = 0.2
boundary_points = np.array([-1.15, -0.4, 0.5, 1.15])
confidence_level = 0.80
uni_controller = create_clf_unicycle_pose_controller()

uni_barrier_cert = create_pr_unicycle_barrier_certificate_cent(safety_radius=safety_radius, Confidence=confidence_level)


'''
v_rand_span = 0.005 * np.ones((2, n)) # setting up velocity error range for each robot
x_rand_span_x = 0.02 * np.random.randint(3, 4, (1, n)) # setting up position error range for each robot,
x_rand_span_y = 0.02 * np.random.randint(1, 4, (1, n)) # rand_span serves as the upper bound of uncertainty for each of the robot
x_rand_span_xy = np.concatenate((x_rand_span_x, x_rand_span_y))
'''

def control_robots(x0,x1,x2,x3, y0,y1,y2,y3, t0,t1,t2,t3 ,goals):
	

    # reshape poses so that its 3 x n
    x = np.array([[x0,x1,x2,x3], [y0,y1,y2,y3], [t0,t1,t2,t3]])
    
	# TODO: Add noise

    # get distance to gaol for all robots
    d = np.sqrt((goals[0][:4] - x[0][:4]) ** 2 + (goals[1][:4] - x[1][:4]) ** 2)

    # stop if distance threshold is met
    if (d < .025).all() or (flag == 1).all():
        V = np.zeros(khep_node_cnt)
        W = np.zeros(khep_node_cnt)
        sub.unregister()
        return V, W

    # Use a position controller to drive to the goal position
    dxu = uni_controller(x, goals)

    dxu = uni_barrier_cert(dxu, x)

    for i in range(khep_node_cnt):
        if round(d[i], 2) < .05 or flag[i] == 1:
            dxu[0, i] = 0
            dxu[1, i] = 0
            flag[i] = 1

    return dxu[0] * 1000, dxu[1]

# This callback function is where the centralized swarm algorithm, or any algorithm should be
# data is the info subscribed from the vicon node, contains the global position, velocity, etc
# the algorithm placed inside this callback should be published to the K4_controls topics
# which should have the K4_controls message type:
# Angular velocity: ctrl_W
# Linear velocity: ctrl_V
def callback(data):
	# arrange positions and goals into PrSBC form
	
	# call control_robots with arranged positions/goals and return V and W list
	
	V, W = control_robots(data.x[0],data.x[1],data.x[2],data.x[3], data.y[0],data.y[1],data.y[2],data.y[3], data.theta[0],data.theta[1],data.theta[2],data.theta[3], uni_goals)

	# send V and W list
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
		pub.get(str(data.ip[i])).publish(control_msgs)

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
