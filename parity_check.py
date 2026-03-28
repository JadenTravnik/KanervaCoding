import time

import gym
import numpy as np
import torch

from agent import QValueAgent
from agent_torch import QValueAgentTorch
from kanerva import BaseKanervaCoder
from kanerva_torch import KanervaLayer


def _make_observation_space() -> gym.spaces.Box:
    return gym.spaces.Box(
        low=np.array([-1.2, -0.07], dtype=np.float32),
        high=np.array([0.6, 0.07], dtype=np.float32),
        dtype=np.float32,
    )


def _assert_close(name: str, lhs: np.ndarray, rhs: np.ndarray, eps: float = 1e-6):
    diff = np.max(np.abs(lhs - rhs))
    if diff > eps:
        raise AssertionError(f"{name} mismatch: max diff={diff}")


def check_feature_parity() -> None:
    observation_space = _make_observation_space()
    n_features = 200
    n_closest = 25

    np.random.seed(123)
    torch.manual_seed(123)

    np_layer = BaseKanervaCoder(observation_space, n_features, n_closest)
    torch_layer = KanervaLayer(observation_space, n_features, n_closest)

    with torch.no_grad():
        torch_layer.prototypes.copy_(torch.from_numpy(np_layer.prototypes).float())

    state = np.array([-0.5, 0.02], dtype=np.float32)

    np_idx = np_layer.get_features(state)
    np_mask = np.zeros(n_features, dtype=np.float32)
    np_mask[np_idx] = 1.0

    torch_state = torch.tensor(state, dtype=torch.float32)
    torch_mask = torch_layer(torch_state).squeeze(0).detach().cpu().numpy()

    _assert_close("feature mask", np_mask, torch_mask)


def check_agent_parity() -> None:
    n_features = 200
    n_actions = 3

    np.random.seed(321)
    torch.manual_seed(321)

    np_agent = QValueAgent(n_features, n_actions, alpha=0.1, epsilon=0.0, gamma=0.99, lmbda=0.99)
    torch_agent = QValueAgentTorch(n_features, n_actions, alpha=0.1, epsilon=0.0, gamma=0.99, lmbda=0.99)

    init_w = np.random.randn(n_features, n_actions).astype(np.float32)
    np_agent.w = init_w.copy()
    with torch.no_grad():
        torch_agent.linear_head.weight.copy_(torch.from_numpy(init_w.T))

    state = np.random.choice(n_features, size=25, replace=False)
    next_state = np.random.choice(n_features, size=25, replace=False)
    action = 1
    reward = -1.0

    np_act = np_agent.act(state, greedy=True)
    torch_act = torch_agent.act(torch.tensor(state, dtype=torch.long), greedy=True)
    if np_act != torch_act:
        raise AssertionError(f"action mismatch: numpy={np_act}, torch={torch_act}")

    np_rollout = np_agent.rollout(next_state)
    torch_rollout = torch_agent.rollout(torch.tensor(next_state, dtype=torch.long))
    if abs(np_rollout - torch_rollout) > 1e-6:
        raise AssertionError(f"rollout mismatch: numpy={np_rollout}, torch={torch_rollout}")

    np_td_err = np_agent.update(state, action, reward, next_state)
    torch_td_err = torch_agent.update(torch.tensor(state, dtype=torch.long), action, reward, torch.tensor(next_state, dtype=torch.long))

    if abs(np_td_err - torch_td_err) > 1e-6:
        raise AssertionError(f"td error mismatch: numpy={np_td_err}, torch={torch_td_err}")

    _assert_close("weights", np_agent.w, torch_agent.w.detach().cpu().numpy(), eps=1e-5)
    _assert_close("eligibility traces", np_agent.e, torch_agent.e.detach().cpu().numpy(), eps=1e-6)


def benchmark_step_runtime(n_steps: int = 10000) -> None:
    n_features = 2000
    n_actions = 3
    n_active = 25

    np.random.seed(42)
    torch.manual_seed(42)

    np_agent = QValueAgent(n_features, n_actions, alpha=0.1, epsilon=0.0, gamma=0.99, lmbda=0.99)
    torch_agent = QValueAgentTorch(n_features, n_actions, alpha=0.1, epsilon=0.0, gamma=0.99, lmbda=0.99)

    init_w = np.random.randn(n_features, n_actions).astype(np.float32)
    np_agent.w = init_w.copy()
    with torch.no_grad():
        torch_agent.linear_head.weight.copy_(torch.from_numpy(init_w.T))

    states = np.random.randint(0, n_features, size=(n_steps, n_active))
    next_states = np.random.randint(0, n_features, size=(n_steps, n_active))
    actions = np.random.randint(0, n_actions, size=n_steps)
    rewards = np.random.randn(n_steps).astype(np.float32)

    start = time.perf_counter()
    for i in range(n_steps):
        np_agent.update(states[i], int(actions[i]), float(rewards[i]), next_states[i])
    numpy_time = time.perf_counter() - start

    states_t = torch.tensor(states, dtype=torch.long)
    next_states_t = torch.tensor(next_states, dtype=torch.long)

    start = time.perf_counter()
    for i in range(n_steps):
        torch_agent.update(states_t[i], int(actions[i]), float(rewards[i]), next_states_t[i])
    torch_time = time.perf_counter() - start

    speedup = numpy_time / torch_time if torch_time > 0 else float("inf")
    print(f"CPU timing over {n_steps} step() updates")
    print(f"NumPy step():   {numpy_time:.6f}s")
    print(f"PyTorch step(): {torch_time:.6f}s")
    print(f"NumPy/PyTorch:  {speedup:.4f}x")


def main() -> None:
    check_feature_parity()
    check_agent_parity()
    benchmark_step_runtime()
    print("Parity check passed")


if __name__ == "__main__":
    main()
