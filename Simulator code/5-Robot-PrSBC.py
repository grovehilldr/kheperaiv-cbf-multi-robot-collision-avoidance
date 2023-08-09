import math
import time
import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse

from scipy.special import comb
from utilities.controllers import *
from utilities.pr_barrier_certs import *
from utilities import util
from utilities.transformations import *

import random



#dxi = np.array([[0, 0, 0, 0,0], [0, 0, 0, 0,0]])

#x = np.array([[0, 0, 0, 0,0], [0, 0, 0, 0,0]])
N = 5
v_rand_span = 0.005 * np.ones((2, N))  # setting up velocity error range for each robot

x_rand_span_x = 30 * np.random.randint(1, 2, (1, N))  # setting up position error range for each robot,
x_rand_span_y = 30 * np.random.randint(1, 2, (1, N))
x_rand_span_xy = np.concatenate((x_rand_span_x, x_rand_span_y))



sim, client = util.start_sim()
agents, targets, n = util.init_robots(sim)
targets = np.array([util.get_target_position(sim, target) for target in targets])
x_goal = targets.T

safety_radius = 0.12 #0.2
confidence_level = 0.51
uni_controller = create_clf_unicycle_pose_controller()
uni_barrier_cert = create_pr_unicycle_barrier_certificate_cent(safety_radius=safety_radius, confidence_level=confidence_level, XRandSpan=x_rand_span_xy, URandSpan=v_rand_span)

loop = True
stopped = np.zeros(len(agents))
while loop:

    # ALGORITHM
    positions = np.array([agent.get_position(sim) for agent in agents])

    x = positions.T.copy()
    print(x)
    d = np.sqrt((x_goal[0] - x[0]) ** 2 + (x_goal[1] - x[1]) ** 2)
    if (d < .05).all() or (stopped == 1).all():
        util.stop_all(sim, agents)
        loop = False
        continue
    dxu = uni_controller(x, x_goal)

    for i in range(len(agents)):
        if round(d[i], 2) < .05 or stopped[i] == 1:
            dxu[0, i] = 0
            dxu[1, i] = 0
            stopped[i] = 1
    dxu = uni_barrier_cert(dxu, x)

    util.set_velocities(sim, agents, dxu)
input('Press any key to continue')
sim.stopSimulation()
