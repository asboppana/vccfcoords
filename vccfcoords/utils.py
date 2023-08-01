import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt

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

#def get_avg_dist_error()


def plot_coordinates(coords: np.ndarray, 
                     ndim: int):
    
    """
    Plot coordindates using matplotlib
    :param coords: coordinates for points to plot
    :param ndim: dimensionality of coordinates
    :raises: ValueError if ndim is not 2 or 3
    """

    if ndim not in [2,3]:
        raise ValueError("ndim must be 2 or 3")

    if ndim == 2:
        x_vals = coords[:, 0:1]
        y_vals = coords[:, 1:2]

        plt.scatter(x_vals, y_vals)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        return
    
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Generate the values
        x_vals = coords[:, 0:1]
        y_vals = coords[:, 1:2]
        z_vals = coords[:, 2:3]

        # Plot the values
        ax.scatter(x_vals, y_vals, z_vals, c = 'b', marker='o')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.show()

        return