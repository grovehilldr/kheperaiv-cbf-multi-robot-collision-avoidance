import numpy as np
def trap_cdf_inv(a, c, delta, sigma):
    # returns list of b2, b1, sigma
    b2 = delta
    b1 = delta

    # a and c should be positive

    if a > c:  # [-A, A] is the large one, and[-C, C] is the smaller one
        A = a
        C = c
    else:
        A = c
        C = a

    if A == 0 and C == 0:
        return b2, b1, sigma

    # O_vec = [-(A + C), -(A - C), (A - C), (A + C)] # vector of vertices on the trap distribution cdf

    h = 1 / (2 * A)  # height of the trap distribution
    area_seq = [1 / 2 * 2 * C * h, 2 * (A - C) * h, 1 / 2 * 2 * C * h]
    area_vec = [area_seq[0], sum(area_seq[:2])]

    if abs(A - C) < 1e-5:  # then is triangle
        # assuming sigma > 50
        b1 = (A + C) - 2 * C * np.sqrt((1 - sigma) / (1 - area_vec[1]))  # 1 - area_vec[1] should be very close to 0.5
        b2 = -b1

        b1 = b1 + delta
        b2 = b2 + delta  # apply shift here due to xi - xj

    else:  # than is trap
        if sigma > area_vec[1]:  # right triangle area
            b1 = (A + C) - 2 * C * np.sqrt((1 - sigma) / (1 - area_vec[1]))
            b2 = -(A + C) + 2 * C * np.sqrt((1 - sigma) / (1 - area_vec[1]))

            b1 = b1 + delta
            b2 = b2 + delta  # apply shift here due to xi - xj

        elif sigma > area_vec[0] and sigma <= area_vec[1]:  # in between the triangle part
            b1 = -(A - C) + (sigma - area_vec[0]) / h  # assuming > 50%, then b1 should > 0
            b2 = -b1

            b1 = b1 + delta
            b2 = b2 + delta  # apply shift here due to xi - xj

            # note that b1 could be > or < b2, depending on whether sigma > or < .5

        elif sigma <= area_vec[0]:
            b1 = -(A + C) + 2 * C * np.sqrt(sigma / area_vec[0])  # assuming > 50%, then b1 should > 0
            b2 = -b1

            b1 = b1 + delta
            b2 = b2 + delta  # apply shift here due to xi - xj

        else:
            print('first triangle, which is not allowed as long as we assume sigma > 50%')

    return b2, b1, sigma