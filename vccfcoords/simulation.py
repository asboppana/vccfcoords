import numpy as np
from scipy.spatial import distance
from vccfcoords import solvers
import numpy.typing as npt
import itertools
from typing import Dict, Any, Tuple, List, Union, Set

def get_euclid_dist(pt1:Union[Tuple[float, float, float], Tuple[float,float]], 
                    pt2:Union[Tuple[float, float, float], Tuple[float,float]]) -> float:
    """
    Euclidean distance between two points
    :param pt1: coordinates of point 1
    :param pt2: coordinates of point 2
    :return: euclidean distance between two points
    """
    
    return distance.euclidean(pt1, pt2)

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
                 n_ref: int,
                 solver_type: str,
                 seed: int=0):
        
        """
        :param n_EC: number of endothelial cells to include in the simulation
        :param n_OC: number of other cells to include in the simulation
        :param n_dim: number of dimension for the simulated and computed coordinates
        :param coord_min: minimum value of the simulated coordinates
        :param coord_max: maximum value of the simulated coordinates
        :param n_ref: number of endothelial reference cells with known coordinates for each other cell of unknown coordinates
        :param solver_type: value that specifies whether to use a geometric solver or an optimization solver (options 'geom' or 'opt')
        :param seed: value that ensures that if all other parameters of the simulation are the same, 
                     then simulations with the same seed value will contain the same simulated points
        """

        #Setting Instance Varibales
        self.n_EC = n_EC
        self.n_OC = n_OC
        self.n_dim = n_dim
        self.coord_min = coord_min
        self.coord_max = coord_max
        self.n_ref = n_ref
        self.solver_type = solver_type
        self.seed = seed
     
        self.n_all = self.n_EC + self.n_OC

        #Setting Seed for reproducibility
        np.random.seed(seed)

        #Simulating Coordinates
        self.coords = Simulation.sim_coordinates(self.n_all,
                                                 self.n_dim,
                                                 self.coord_min,
                                                 self.coord_max)
        
        #Setting EC and OC indices 
        self.EC_idx = list(set(np.random.choice(list(range(self.n_all)), self.n_EC, replace = False)))
        self.OC_idx = list(set(list(range(self.n_all))) - set(self.EC_idx))

        #Generating Closest Points Dictionary
        self.closest_points_dict = self.gen_closest_points_dict()

        #Estimating OC coordinates
        self.OC_coords_pred = self.solve()

        #Computing Metrics
        self.avg_cell_dist = self.get_avg_cell_dist()

    @staticmethod
    def sim_coordinates(n: int, 
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
        
        return generated_coordinates

    @staticmethod
    def get_closest_points(coordinate_pool: np.ndarray,
                           coordinate_target: np.ndarray,
                           n_closest: int) -> Tuple[List[int], List[float]]:
        
        """
        Get the n closest points given a target point
        :param coordinate_pool: the pool of points to search
        :param coordinate_target: the coordinates of the target point 
        :param n_closest: the number of closest points to find
        :return: the indices of the n closest points and the distances to those points
        """

        # Calculate distances from target coordinate to each point in the pool
        distances = np.linalg.norm(coordinate_pool - coordinate_target, axis=1)

        # Sort indices based on distances and get the n closest points
        closest_indices = np.argsort(distances)[:n_closest]
        closest_distances = list(distances[closest_indices])

        return closest_distances, closest_indices

    def gen_closest_points_dict(self) -> Dict[int, Tuple[List[int], List[float]]]:

        """
        Get the n closest points and distances from a list of target points
        return: dictionary of OC indices as keys and tuples of closest EC indices and distance lists as values
        """

        #EC coordinate pool
        pool_coords = self.coords[self.EC_idx]

        #Dictionary to convert sequential indices to EC indices
        convert_idx_dict = dict(zip(list(range(self.n_EC)), self.EC_idx))

        closest_points_dict = dict()

        #Generating closest point indices and distances for each OC cell
        for idx in self.OC_idx:
            target_coord = self.coords[idx]
            closest_distances, closest_points_idx = Simulation.get_closest_points(pool_coords, 
                                                                                  target_coord, 
                                                                                  self.n_ref)
            
            closest_points_idx_converted = [convert_idx_dict[x] for x in closest_points_idx]
            
            closest_points_dict[idx] = (closest_distances, closest_points_idx_converted)

        return closest_points_dict
    
    def solve(self) -> Dict[int, np.ndarray]:

        """
        Applies a 2D or 3D, geometric or optimization solver function to the tri/multi-lateration problem.
        Specifically, this function computes or estimates the OC coordinates given distances to EC coordinates.
        return: dictionary of OC indices as keys and a numpy array of estimated coordinates as values.
        """

        solver_func = None
        DIST_IDX = 0
        POINTS_IDX = 1

        if self.n_dim == 2 and self.solver_type == "geom":
            solver_func = solvers.geom_2D_solver
        if self.n_dim == 2 and self.solver_type == "opt":
            solver_func = solvers.opt_lse_2D_solver
        if self.n_dim == 3 and self.solver_type == "geom":
            solver_func = solvers.geom_3D_solver
        if self.n_dim == 3 and self.solver_type == "opt":
            solver_func = solvers.opt_lse_3D_solver

        estimated_coords_dict = {x: np.asarray(solver_func(y[DIST_IDX], 
                                                self.coords[y[POINTS_IDX]]))
                                                for x,y in self.closest_points_dict.items()}

        return estimated_coords_dict 
    
    def get_avg_cell_dist(self):

        return [np.linalg.norm(x[0]-x[1]) for x in itertools.combinations(self.coords, 2)]
        



        


         

       

            


