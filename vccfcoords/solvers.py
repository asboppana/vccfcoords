import numpy as np
from scipy.spatial import distance
from typing import Dict, Any, Tuple, List, Union


def geom_2D_solver(distances: List[float],
                   points: List[Tuple[float, float]]) -> Tuple[float, float]:

    """
    Geometric solver to the trilateration problem in 2 dimensions. 
    :param distances: distances from an unknown point to three reference points with known coordinates
    :param points: coordinates of the three reference points
    :return: computed coordinates of the unknown point
    """
    
    POINT1_IDX = 0
    POINT2_IDX = 1
    POINT3_IDX = 2

    x1, y1 = points[POINT1_IDX]
    x2, y2 = points[POINT2_IDX]
    x3, y3 = points[POINT3_IDX]
    r1, r2, r3 = distances
    
    A = 2 * (x2 - x1)
    B = 2 * (y2 - y1)
    C = r1**2 - r2**2 - x1**2 + x2**2 - y1**2 + y2**2
    D = 2 * (x3 - x2)
    E = 2 * (y3 - y2)
    F = r2**2 - r3**2 - x2**2 + x3**2 - y2**2 + y3**2

    x = (C*E - F*B) / (E*A - B*D)
    y = (C*D - A*F) / (B*D - A*E)

    return x,y

def geom_3D_solver(distances: List[float], 
                   points: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    
    """
    Geometric solver to the multilateration problem in 3 dimensions.
    :param distances: distances from an unknown point to four reference points with known coordinates
    :param points: coordinates of the four reference points
    :return: computed coordinates of the unknown point
    :raises ValueError: If the points provided do not form a tetrahedron, leading to no unique solution.
    """

    POINT1_IDX = 0
    POINT2_IDX = 1
    POINT3_IDX = 2
    POINT4_IDX = 3

    x1, y1, z1 = points[POINT1_IDX]
    x2, y2, z2 = points[POINT2_IDX]
    x3, y3, z3 = points[POINT3_IDX]
    x4, y4, z4 = points[POINT4_IDX]
    r1, r2, r3, r4 = distances
    
    # Calculate the coefficients for the linear equations
    A = x2 - x1
    B = y2 - y1
    C = z2 - z1
    D = 0.5 * (r1**2 - r2**2 + x2**2 - x1**2 + y2**2 - y1**2 + z2**2 - z1**2)
    E = x3 - x1
    F = y3 - y1
    G = z3 - z1
    H = 0.5 * (r1**2 - r3**2 + x3**2 - x1**2 + y3**2 - y1**2 + z3**2 - z1**2)
    I = x4 - x1
    J = y4 - y1
    K = z4 - z1
    L = 0.5 * (r1**2 - r4**2 + x4**2 - x1**2 + y4**2 - y1**2 + z4**2 - z1**2)

    # Set up the 3x3 matrix and the right-hand side vector
    M = np.array([[A, B, C], [E, F, G], [I, J, K]])
    N = np.array([D, H, L])

    try:
        # Solve the system of linear equations to find the x, y, and z coordinates
        x, y, z = np.linalg.solve(M, N)
    except np.linalg.LinAlgError:
        raise ValueError("No unique solution. The points may not form a tetrahedron.")
    
    return x, y, z

