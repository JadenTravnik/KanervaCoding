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

        :param data: data to normalize
        :return: normalized data
        """
        return (data - self.obs_low) / self.obs_range

    def distance(self, data: torch.Tensor) -> torch.Tensor:
        """
        Computes the distance between the data and the prototypes.
        Defaults to euclidian distance

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
