import numpy as np
from scipy.spatial import distance
from typing import Dict, Any, Tuple, List, Union

def sim_coordinates(n: int, 
                    ndim: int, 
                    min_val:float=0,
                    max_val:float=1) -> np.ndarray:
    """
    Randomly simulate a number of point coordinates in a specified dimension between min and max values
    :param n: number of points to be simulated
    :param ndim: dimensionality of coordinates
    :param min_val: minimum value of simulated coordinates
    :param max_val: maximum value of simulated coordinates
    :return: matrix of simulated coordinates
    """

    val_range = max_val - min_val
    generated_coordinates = np.random.rand(n, ndim) * val_range + min_val
    
    return generated_coordinates

def get_euclid_dist(pt1:Union[Tuple[float, float, float], Tuple[float,float]], 
                    pt2:Union[Tuple[float, float, float], Tuple[float,float]]) -> float:
    """
    Euclidean distance between two points
    :param pt1: coordinates of point 1
    :param pt2: coordinates of point 2
    :return: euclidean distance between two points
    """
    
    return distance.euclidean(pt1, pt2)

def get_closest_points(coords: np.ndarray, 
                       target: Union[Tuple[float, float, float], Tuple[float,float]],
                       n: int) -> Tuple[List[int], List[int]]:
    """
    Get the n closest points given a target point
    :param coords: the pool of points to search
    :param target: the coordinates of the target point 
    :param n: the number of closest points to find
    :return: the indices of the n closest points and the distances to those points
    """

    DIST_IDX = 0

    distances = distance.cdist([target], coords, metric='euclidean')[DIST_IDX]

    closest_idx = list(np.argsort(distances)[:n])
    closest_distances = list(distances[closest_idx])

    return closest_idx, closest_distances