import numpy as np
from typing import NoReturn

class QValueAgent:
    
    def __init__(self, n_features: int, n_actions: int, alpha: float=0.01, epsilon: float=0.1, gamma: float=0.9, lmbda: float=0.9):
        """
        Simple Q Learning Agent with elegability traces

        :param n_features: number of features that represent the space
        :param n_actions: number of discrete actions
        :param alpha: learning rate
        :param epsilon: epsilon greedy policy
        :param gamma: discount factor
        :param lmbda: elegibility trace factor
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.w = np.zeros((n_features, n_actions))
        self.e = np.zeros(n_features)
        
    def act(self, state: np.ndarray, greedy: bool=False) -> int:
        """
        Computes an action using epsilon-greedy

        :param state: indicies of active features
        :param greedy: if policy should be greedy

        :return: action index
        """
        if greedy or np.random.random() > self.epsilon:
            q = self.w[state].sum(axis=0)
            action = q.argmax()
        else:
            action = np.random.choice(self.n_actions)

        return action

    def rollout(self, state: np.ndarray) -> float:
        """
        Gets the value of rolling out the state 

        :param state: indicies of active features
        :return: rollout value
        """
        return self.gamma * self.w[state].sum(axis=0).max()

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> float:
        """
        Updates the agent using the qlearning with elegabilty traces update

        :param state: indicies of active features for the state
        :param action: action index
        :param reward: reward of the transition
        :param next_state: indicies of active features for the next state
        :return: TD error
        """
        # update traces
        self.e *= self.lmbda * self.gamma
        self.e[state] = 1

        # get v_next from rollout
        v_next = self.rollout(next_state)

        # calculate td_error based on current value 
        td_err = reward + v_next - self.w[state, action].sum()

        # the len(state) is ensure that the total update is distributed over the feature weights
        alpha = self.alpha / len(state)

        # update weights
        self.w[:, action] += alpha * td_err * self.e

        return td_err

    def erase_traces(self) -> NoReturn:
        """
        Erases the traces
        """
        self.e *= 0

    def save(self, filename: str) -> NoReturn:
        """
        Saves the weights of the agent to an npy file
        """
        np.save(filename, self.w)

    def load(self, filename):
        """
        Loads the weights of the agent stored in an npy file
        """
        self.w = np.load(filename)



