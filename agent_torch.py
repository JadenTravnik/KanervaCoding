import torch
from torch import nn


class QValueAgentTorch(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        alpha: float = 0.01,
        epsilon: float = 0.1,
        gamma: float = 0.9,
        lmbda: float = 0.9,
        device: torch.device | None = None,
    ):
        """
        Simple Q Learning Agent with eligibility traces implemented in PyTorch.

        :param n_features: number of features that represent the space
        :param n_actions: number of discrete actions
        :param alpha: learning rate
        :param epsilon: epsilon greedy policy
        :param gamma: discount factor
        :param lmbda: eligibility trace factor
        :param device: device to run the model on
        """
        super().__init__()
        self.n_actions = n_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.lmbda = lmbda

        if device is None:
            device = torch.device("cpu")

        self.linear_head = nn.Linear(n_features, n_actions, bias=False, device=device)
        with torch.no_grad():
            self.linear_head.weight.zero_()

        self.register_buffer("e", torch.zeros(n_features, dtype=torch.float32, device=device))

    @property
    def w(self) -> torch.Tensor:
        """
        Returns weight matrix with shape (n_features, n_actions) to mirror NumPy version.
        """
        return self.linear_head.weight.t()

    def q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Gets action values for a state represented by feature indices or dense feature vector.

        :param state: active feature indices or dense binary feature vector
        :return: q values for each action
        """
        if state.dtype in (torch.int32, torch.int64):
            return self.w[state].sum(dim=0)

        return self.linear_head(state.float())

    def act(self, state: torch.Tensor, greedy: bool = False) -> int:
        """
        Computes an action using epsilon-greedy

        :param state: indices of active features or dense features
        :param greedy: if policy should be greedy

        :return: action index
        """
        if greedy or torch.rand(1).item() > self.epsilon:
            q = self.q_values(state)
            action = int(torch.argmax(q).item())
        else:
            action = int(torch.randint(low=0, high=self.n_actions, size=(1,)).item())

        return action

    def rollout(self, state: torch.Tensor) -> float:
        """
        Gets the value of rolling out the state

        :param state: indices of active features or dense features
        :return: rollout value
        """
        return float(self.gamma * self.q_values(state).max().item())

    def update(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor) -> float:
        """
        Updates the agent using the qlearning with eligibility traces update

        :param state: indices of active features for the state
        :param action: action index
        :param reward: reward of the transition
        :param next_state: indices of active features for the next state
        :return: TD error
        """
        with torch.no_grad():
            # update traces
            self.e *= self.lmbda * self.gamma
            self.e[state] = 1.0

            # get v_next from rollout
            v_next = self.rollout(next_state)

            # calculate td_error based on current value
            td_err = reward + v_next - float(self.w[state, action].sum().item())

            # normalize by active features so effective learning rate remains stable as sparsity changes
            alpha = self.alpha / len(state)

            # update weights
            self.linear_head.weight[action] += alpha * td_err * self.e

        return td_err

    def erase_traces(self) -> None:
        """
        Erases the traces
        """
        self.e *= 0

    def save(self, filename: str) -> None:
        """
        Saves the weights of the agent to a torch file
        """
        torch.save(self.state_dict(), filename)

    def load(self, filename: str) -> None:
        """
        Loads the weights of the agent stored in a torch file
        """
        self.load_state_dict(torch.load(filename, map_location=self.e.device))
