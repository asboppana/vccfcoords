import numpy as np
import itertools
from scipy.spatial import distance
from scipy.optimize import least_squares
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

def geom_2D_comb_solver(distances: List[float], 
                        points: List[Tuple[float, float]]) -> Tuple[float, float]:
    
    """
    Geometric solver to the multilateration problem in 2 dimensions that averages calculations from
    each combination of 3 points
    :param distances: distances from an unknown point to at least three reference points with known coordinates
    :param points: coordinates of at least three reference points
    :return: average computed coordinates of the unknown point
    """
    
    GEOM_MAX_2D = 3
    X_IDX = 0
    Y_IDX = 1

    combination_idx = [list(x) for x in list(itertools.combinations(range(len(distances)), GEOM_MAX_2D))]

    pred_coords_list = [geom_2D_solver(np.array(distances)[idx], points[idx]) for idx in combination_idx]

    x = np.mean([coord[X_IDX] for coord in pred_coords_list])
    y = np.mean([coord[Y_IDX] for coord in pred_coords_list])

    return x,y

def geom_3D_comb_solver(distances: List[float], 
                        points: List[Tuple[float, float]]) -> Tuple[float, float]:
    
    """
    Geometric solver to the multilateration problem in 3 dimensions that averages calculations from
    each combination of 4 points
    :param distances: distances from an unknown point to at least four reference points with known coordinates
    :param points: coordinates of at least four reference points
    :return: average computed coordinates of the unknown point
    :raises ValueError: If the points provided do not form a tetrahedron, leading to no unique solution.
    """
    
    GEOM_MAX_3D = 4
    X_IDX = 0
    Y_IDX = 1
    Z_IDX = 2

    combination_idx = [list(x) for x in list(itertools.combinations(range(len(distances)), GEOM_MAX_3D))]

    pred_coords_list = [geom_3D_solver(np.array(distances)[idx], points[idx]) for idx in combination_idx]

    x = np.mean([coord[X_IDX] for coord in pred_coords_list])
    y = np.mean([coord[Y_IDX] for coord in pred_coords_list])
    z = np.mean([coord[Z_IDX] for coord in pred_coords_list])

    return x,y,z
    


def opt_lse_2D_solver(distances: List[float], 
                      points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Least-square-error optimization solver to the multilateration problem in 2 dimensions.
    :param distances: distances from an unknown point to at least three reference points with known coordinates
    :param points: coordinates of at least three reference points
    :return: computed coordinates of the unknown point
    :raises ValueError: If number of distances and points are not the same, or less then 3 reference points are given
    """

    if len(distances) != len(points) or len(distances) < 3:
        raise ValueError("Number of distances and points must be the same, and at least 3 reference points are required.")

   
    # Prepare data
    x_coords, y_coords = zip(*points)
    p_ref = np.array(points)

    # Calculate centroid of reference points
    x0 = np.mean(x_coords)
    y0 = np.mean(y_coords)

    # Error function for optimization
    def error_func(p):
        x, y = p
        return [(x - x_ref)**2 + (y - y_ref)**2 - dist**2 for (x_ref, y_ref), dist in zip(p_ref, distances)]

    # Optimize using least squares with "lm" method and better initial guess
    result = least_squares(error_func, [x0, y0], method='lm')

    x, y = result.x
    return x, y

def opt_lse_3D_solver(distances: List[float], points: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """
    Least-square-error optimization solver to the multilateration problem in 3 dimensions.
    :param distances: distances from an unknown point to four or more reference points with known coordinates
    :param points: coordinates of the four or more reference points
    :return: computed coordinates of the unknown point
    :raises ValueError: If number of distances and points are not the same, or less than four reference points are given
    """

    if len(distances) != len(points) or len(distances) < 4:
        raise ValueError("Number of distances and points must be the same, and at least four reference points are required.")

    # Prepare data
    x_coords, y_coords, z_coords = zip(*points)
    p_ref = np.array(points)

    # Calculate centroid of reference points
    x0 = np.mean(x_coords)
    y0 = np.mean(y_coords)
    z0 = np.mean(z_coords)

    # Error function for optimization
    def error_func(p):
        x, y, z = p
        return [(x - x_ref)**2 + (y - y_ref)**2 + (z - z_ref)**2 - dist**2 for (x_ref, y_ref, z_ref), dist in zip(p_ref, distances)]

    # Optimize using least squares with "lm" method and better initial guess
    result = least_squares(error_func, [x0, y0, z0], method='lm')

    x, y, z = result.x
    return x, y, z


