import numpy as np
from scipy.spatial import distance
from typing import Dict, Any, Tuple, List, Union

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

class Simulation:
    """
    A :class:`Simulation` is a simulation object which contains the state of simulated point coordinates, 
    and has operations to call various coordinate solvers, evaluate performances of a solver, 
    and add noise to simulated coordinates.
    """

    def __init__(self, 
                 n_EC: int,
                 n_OC: int,
                 n_dim: int,
                 coord_min: float,
                 coord_max: float,
                 seed: int=0):
        
        """
        :param n_EC: number of endothelial cells to include in the simulation
        :param n_OC: number of other cells to include in the simulation
        :param n_dim: number of dimension for the simulated and computed coordinates
        :param coord_min: minimum value of the simulated coordinates
        :param coord_max: maximum value of the simulated coordinates
        :param seed: value that ensures that if all other parameters of the simulation are the same, 
                     then simulations with the same seed value will contain the same simulated points
        """
        self.n_EC = n_EC
        self.n_OC = n_OC
        self.n_dim = n_dim
        self.coord_min = coord_min
        self.coord_max = coord_max
        self.seed = seed

        np.random.seed(seed)

    def sim_coordinates(self,
                        n: int, 
                        n_dim: int, 
                        min_val: float=0,
                        max_val: float=1) -> np.ndarray:
        """
        Randomly simulate a number of point coordinates in a specified dimension between min and max values
        :param n: number of points to be simulated
        :param n_dim: dimensionality of coordinates
        :param min_val: minimum value of simulated coordinates
        :param max_val: maximum value of simulated coordinates
        :return: matrix of simulated coordinates
        """

        val_range = max_val - min_val
        generated_coordinates = np.random.rand(n, n_dim) * val_range + min_val
        
        return generated_coordinate

    def method2(self, new_text):
        """
        A method to modify attribute2.

        Args:
            new_text (str): The new string value to set attribute2.
        """
        self.attribute2 = new_text
