import numpy as np
import gym

class BaseKanervaCoder:
    def __init__(self, observation_space: gym.spaces.Space, n_prototypes: int, n_closest: int):
        """
        Base Kanerva Coder using Selective Kanerva Coding

        :param observation_space: space to approximate
        :param n_prototypes: number of prototypes to represent space
        :param n_closest: number of active prototypes
        """
        self.n_prototypes = n_prototypes
        self.dimensions = observation_space.low.shape[0]
        self.observation_space = observation_space
        self.observation_range = observation_space.high - observation_space.low
        self.prototypes = np.random.rand(n_prototypes, self.dimensions)
        self.visit_counts = np.zeros(n_prototypes)
        self.n_closest = n_closest

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalizes the data to be between 0,1

        :param data: data to normalize
        :return: normalized data
        """
        normed_data = data - self.observation_space.low
        normed_data /= self.observation_range
        return normed_data

    def distance(self, data: np.ndarray) -> np.ndarray:
        """
        Computes the distance between the data and the prototypes.
        Defaults to euclidian distance

        :param data:
        :return: array of distance values between the input data and each prototype
        """
        data = self.normalize(data)
        dist = self.prototypes - data
        dist = np.sqrt(sum(dist.T**2))
        return dist

    def get_features(self, data: np.ndarray) -> np.ndarray:
        """
        Gets the active features for the input data. Updates the visit counts

        :param data: input data
        :return: array of active feature indexes
        """
        dist = self.distance(data)
        indexes = np.argpartition(self.distance(data), self.n_closest, axis=0)[:self.n_closest]
        self.visit_counts[indexes] += 1
        return indexes


