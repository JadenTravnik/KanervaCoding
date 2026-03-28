import gym
import torch
from torch import nn


class KanervaLayer(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        n_prototypes: int,
        n_closest: int,
    ):
        """
        PyTorch Kanerva coding layer using selective activation.

        :param observation_space: space to approximate
        :param n_prototypes: number of prototypes to represent space
        :param n_closest: number of active prototypes
        """
        super().__init__()
        self.n_prototypes = n_prototypes
        self.n_closest = n_closest
        self.dimensions = observation_space.low.shape[0]

        low = torch.as_tensor(observation_space.low, dtype=torch.float32)
        high = torch.as_tensor(observation_space.high, dtype=torch.float32)
        self.register_buffer("obs_low", low)
        self.register_buffer("obs_range", high - low)

        prototypes = torch.rand(n_prototypes, self.dimensions, dtype=torch.float32)
        self.prototypes = nn.Parameter(prototypes, requires_grad=False)

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the data to be between 0,1

        :param data: data to normalize (expected within observation space bounds)
        :return: normalized data
        """
        return (data - self.obs_low) / self.obs_range

    def distance(self, data: torch.Tensor) -> torch.Tensor:
        """
        Computes the distance between the data and the prototypes.
        Defaults to euclidean distance

        :param data: normalized input batch
        :return: distance tensor for each prototype
        """
        normed_data = self.normalize(data)
        dist = self.prototypes.unsqueeze(0) - normed_data.unsqueeze(1)
        return torch.linalg.norm(dist, dim=-1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Gets the active feature mask for the input data.

        :param data: input tensor of shape (dimensions,) or (batch, dimensions)
        :return: binary activation tensor of shape (batch, n_prototypes)
        """
        if data.dim() == 1:
            data = data.unsqueeze(0)

        dist = self.distance(data)
        _, indexes = torch.topk(dist, k=self.n_closest, dim=1, largest=False)

        features = torch.zeros(
            data.shape[0],
            self.n_prototypes,
            device=data.device,
            dtype=data.dtype,
        )
        features.scatter_(1, indexes, 1.0)
        return features


class KanervaBinary(nn.Module):
    def __init__(self, n_input_features: int, n_prototypes: int, n_closest: int):
        """
        Binary Kanerva coding layer using Hamming distance and selective activation.

        :param n_input_features: size of incoming binary feature vector
        :param n_prototypes: number of binary prototypes to represent space
        :param n_closest: number of active prototypes
        """
        super().__init__()
        self.n_input_features = n_input_features
        self.n_prototypes = n_prototypes
        self.n_closest = n_closest

        prototypes = torch.randint(
            low=0,
            high=2,
            size=(n_prototypes, n_input_features),
            dtype=torch.bool,
        )
        self.prototypes = nn.Parameter(prototypes, requires_grad=False)

    def distance(self, data: torch.Tensor) -> torch.Tensor:
        """
        Computes Hamming distance between binary input features and binary prototypes.

        :param data: binary input batch
        :return: distance tensor for each prototype
        """
        bin_data = data > 0
        diff = torch.logical_xor(self.prototypes.unsqueeze(0), bin_data.unsqueeze(1))
        return diff.sum(dim=-1).float()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Gets active binary prototypes for the input binary features.

        :param data: input tensor of shape (n_input_features,) or (batch, n_input_features)
        :return: binary activation tensor of shape (batch, n_prototypes)
        """
        if data.dim() == 1:
            data = data.unsqueeze(0)

        dist = self.distance(data)
        _, indexes = torch.topk(dist, k=self.n_closest, dim=1, largest=False)

        features = torch.zeros(
            data.shape[0],
            self.n_prototypes,
            device=data.device,
            dtype=torch.float32,
        )
        features.scatter_(1, indexes, 1.0)
        return features
