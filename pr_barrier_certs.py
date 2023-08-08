from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse

from scipy.special import comb
from utilities.transformations import *
import math
import time
import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse

from scipy.special import comb
from utilities.trap_cdf_inv import *
#from utilities.Telemetry import Telemetry

# Disable output of CVXOPT
options['show_progress'] = False
# Change default options of CVXOPT for faster solving
options['reltol'] = 1e-2 # was e-2
options['feastol'] = 1e-2 # was e-4
options['maxiters'] = 50 # default is 100

N = 5
v_rand_span = 0.005 * np.ones((2, N))  # setting up velocity error range for each robot

x_rand_span_x = 1 * np.random.randint(3, 4, (1, N))  # setting up position error range for each robot,
x_rand_span_y = 1 * np.random.randint(1, 4, (1, N))
x_rand_span_xy = np.concatenate((x_rand_span_x, x_rand_span_y))


def create_si_pr_barrier_certificate_centralized(gamma=100, safety_radius=0.2, magnitude_limit=0.2, confidence_level=1.0,
                                                 XRandSpan=x_rand_span_xy, URandSpan=v_rand_span):

    def barrier_certificate(dxi, x, XRandSpan, URandSpan):
        print(XRandSpan)
        # N = dxi.shape[1]
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
                b2_x, b1_x, sigma = trap_cdf_inv(XRandSpan[0, i], XRandSpan[0, j], x[0, i] - x[0, j], confidence_level)
                b2_y, b1_y, sigma = trap_cdf_inv(XRandSpan[1, i], XRandSpan[1, j], x[1, i] - x[1, j], confidence_level)

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

                t1 = np.array([[b_x], [0.0]])
                t2 = np.array([[max_dvij_x], [0]])
                t3 = np.array([[max_dxij_x], [0]])
                t4 = np.array([[0], [b_y]])
                t5 = np.array([[0], [max_dvij_y]])
                t6 = np.array([[0], [max_dxij_y]])


                h1 = np.linalg.norm(t1) ** 2 - safety_radius ** 2 - 2 * np.linalg.norm(
                    t2) * np.linalg.norm(t3) / gamma
                h2 = np.linalg.norm(t4) ** 2 - safety_radius ** 2 - 2 * np.linalg.norm(
                    t5) * np.linalg.norm(t6) / gamma  # h_y

                h = h1 + h2

                b[count] = gamma * h ** 3  # matlab original: b(count) = gamma*h^3
                count += 1

        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, 0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] = dxi[:, idxs_to_normalize] * (magnitude_limit / norms[idxs_to_normalize])

        f_mat = -2 * np.reshape(dxi, 2 * N, order='F')
        f_mat = f_mat.astype('float')
        result = qp(H, matrix(f_mat), matrix(A), matrix(b))['x']

        return np.reshape(result, (2, -1), order='F')

    return barrier_certificate


def create_pr_unicycle_barrier_certificate_cent(barrier_gain=100, safety_radius=0.12, projection_distance=0.05,
                                                magnitude_limit=0.2, confidence_level=1.0, XRandSpan=x_rand_span_xy, URandSpan=v_rand_span):
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

    # Check user input types
    assert isinstance(barrier_gain, (int,
                                     float)), "In the function create_pr_unicycle_barrier_certificate, the barrier gain (barrier_gain) must be an integer or float. Recieved type %r." % type(
        barrier_gain).__name__
    assert isinstance(safety_radius, (int,
                                      float)), "In the function create_pr_unicycle_barrier_certificate, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(
        safety_radius).__name__
    assert isinstance(projection_distance, (int,
                                            float)), "In the function create_pr_unicycle_barrier_certificate, the projected point distance for the diffeomorphism between sinlge integrator and unicycle (projection_distance) must be an integer or float. Recieved type %r." % type(
        projection_distance).__name__
    assert isinstance(magnitude_limit, (int,
                                        float)), "In the function create_pr_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(
        magnitude_limit).__name__
    assert isinstance(confidence_level,
                      float), "In the function create_pr_unicycle_barrier_certificate, the confidence level must be a float. Recieved type %r." % type(
        confidence_level).__name__

    # Check user input ranges/sizes

    assert barrier_gain > 0, "In the function create_pr_unicycle_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert safety_radius >= 0.12, "In the function create_pr_unicycle_barrier_certificate, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m). Recieved %r." % safety_radius
    assert projection_distance > 0, "In the function create_pr_unicycle_barrier_certificate, the projected point distance for the diffeomorphism between sinlge integrator and unicycle (projection_distance) must be positive. Recieved %r." % projection_distance
    assert magnitude_limit > 0, "In the function create_pr_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_pr_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit
    assert confidence_level <= 1, "In the function create_pr_unicycle_barrier_certificate, the confidence level must be less than 1. Recieved %r." % confidence_level
    assert confidence_level >= 0, "In the function create_pr_unicycle_barrier_certificate, the confidence level must be positive (greater than 0). Recieved %r." % confidence_level

    si_barrier_cert = create_si_pr_barrier_certificate_centralized(gamma=barrier_gain,
                                                                   safety_radius=safety_radius + projection_distance,
                                                                   confidence_level=confidence_level, XRandSpan=x_rand_span_xy,
                                                                   URandSpan=v_rand_span)

    si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping(projection_distance=projection_distance)

    uni_to_si_dyn = create_uni_to_si_dynamics(projection_distance=projection_distance)

    def f(dxu, x, XRandSpan=x_rand_span_xy, URandSpan=v_rand_span):



        # Check user input types
        assert isinstance(dxu,
                          np.ndarray), "In the function created by the create_unicycle_barrier_certificate function, the unicycle robot velocity command (dxu) must be a numpy array. Recieved type %r." % type(
            dxu).__name__
        assert isinstance(x,
                          np.ndarray), "In the function created by the create_unicycle_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(
            x).__name__

        # Check user input ranges/sizes
        assert x.shape[
                   0] == 3, "In the function created by the create_unicycle_barrier_certificate function, the dimension of the unicycle robot states (x) must be 3 ([x;y;theta]). Recieved dimension %r." % \
                            x.shape[0]
        assert dxu.shape[
                   0] == 2, "In the function created by the create_unicycle_barrier_certificate function, the dimension of the robot unicycle velocity command (dxu) must be 2 ([v;w]). Recieved dimension %r." % \
                            dxu.shape[0]
        assert x.shape[1] == dxu.shape[
            1], "In the function created by the create_unicycle_barrier_certificate function, the number of robot states (x) must be equal to the number of robot unicycle velocity commands (dxu). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (
        x.shape[0], x.shape[1], dxu.shape[0], dxu.shape[1])

        x_si = uni_to_si_states(x)
        # Convert unicycle control command to single integrator one
        dxi = uni_to_si_dyn(dxu, x)
        # Apply single integrator barrier certificate
        dxi = si_barrier_cert(dxi, x_si, XRandSpan, URandSpan)
        # Return safe unicycle command
        return si_to_uni_dyn(dxi, x)

    return f