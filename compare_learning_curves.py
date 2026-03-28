import argparse
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import QValueAgent
from agent_torch import QValueAgentTorch, QValueAgentTorchBinary
from kanerva import BaseKanervaCoder
from kanerva_torch import KanervaBinary, KanervaLayer

NEAR_INF_MULTIPLIER = 0.5
EARLY_WEIGHT_START = 1.0
EARLY_WEIGHT_END = 0.1
TORCH_SEED_MULTIPLIER = 17
TORCH_BINARY_SEED_MULTIPLIER = 23
TUNER_BINARY_SAMPLER_SEED_OFFSET = 1009
SEED_MODULUS = 2**31
# Early episodes get larger weights so the tuning objective favors faster learning.
# Later episodes still contribute, but with smaller weight.
_EARLY_RETURN_WEIGHTS_CACHE: dict[int, np.ndarray] = {}

# gym==0.26 expects np.bool8, which is removed in NumPy 2.x.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


def _reset_env(env: gym.Env, seed: int | None = None):
    reset_out = env.reset(seed=seed)
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out


def _step_env(env: gym.Env, action: int):
    step_out = env.step(action)
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = terminated or truncated
    else:
        obs, reward, done, info = step_out
    return obs, reward, done, info


def _kanerva_observation_space(env: gym.Env, non_finite_bound: float) -> gym.spaces.Box:
    observation_space = env.observation_space
    if not isinstance(observation_space, gym.spaces.Box):
        raise ValueError("Kanerva comparisons require a Box observation space")

    low = np.asarray(observation_space.low, dtype=np.float32)
    high = np.asarray(observation_space.high, dtype=np.float32)

    # Some envs use extreme finite sentinels (e.g., +/-3.4e38) instead of true inf.
    # Treat values above this threshold as effectively unbounded for stable normalization.
    near_inf_threshold = np.finfo(np.float32).max * NEAR_INF_MULTIPLIER
    low = np.where(np.isfinite(low) & (np.abs(low) < near_inf_threshold), low, -non_finite_bound)
    high = np.where(np.isfinite(high) & (np.abs(high) < near_inf_threshold), high, non_finite_bound)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


def train_numpy_agent(
    env_id: str,
    n_episodes: int,
    max_steps: int,
    n_features: int,
    n_closest: int,
    alpha: float,
    epsilon: float,
    gamma: float,
    lmbda: float,
    seed: int,
    non_finite_bound: float,
) -> np.ndarray:
    env = gym.make(env_id)
    env.action_space.seed(seed)

    rep_space = _kanerva_observation_space(env, non_finite_bound)
    rep = BaseKanervaCoder(rep_space, n_features, n_closest)
    agent = QValueAgent(n_features, env.action_space.n, alpha=alpha, epsilon=epsilon, gamma=gamma, lmbda=lmbda)

    returns = np.zeros(n_episodes, dtype=np.float32)
    for i_episode in range(n_episodes):
        observation = _reset_env(env, seed=seed + i_episode)
        old_features = rep.get_features(observation)

        ep_return = 0.0
        for _ in range(max_steps):
            action = agent.act(old_features)
            new_observation, reward, done, _ = _step_env(env, action)
            new_features = rep.get_features(new_observation)
            agent.update(old_features, action, reward, new_features)
            ep_return += reward

            if done:
                agent.erase_traces()
                break

            old_features = new_features

        returns[i_episode] = ep_return

    env.close()
    return returns


def train_torch_agent(
    env_id: str,
    n_episodes: int,
    max_steps: int,
    n_features: int,
    n_closest: int,
    alpha: float,
    epsilon: float,
    gamma: float,
    lmbda: float,
    seed: int,
    non_finite_bound: float,
) -> np.ndarray:
    env = gym.make(env_id)
    env.action_space.seed(seed)

    rep_space = _kanerva_observation_space(env, non_finite_bound)
    rep = KanervaLayer(rep_space, n_features, n_closest)
    agent = QValueAgentTorch(n_features, env.action_space.n, alpha=alpha, epsilon=epsilon, gamma=gamma, lmbda=lmbda)

    returns = np.zeros(n_episodes, dtype=np.float32)
    for i_episode in range(n_episodes):
        observation = _reset_env(env, seed=seed + i_episode)
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        old_features = torch.nonzero(rep(obs_tensor).squeeze(0), as_tuple=False).squeeze(-1)

        ep_return = 0.0
        for _ in range(max_steps):
            action = agent.act(old_features)
            new_observation, reward, done, _ = _step_env(env, action)
            new_obs_tensor = torch.tensor(new_observation, dtype=torch.float32)
            new_features = torch.nonzero(rep(new_obs_tensor).squeeze(0), as_tuple=False).squeeze(-1)
            agent.update(old_features, action, reward, new_features)
            ep_return += reward

            if done:
                agent.erase_traces()
                break

            old_features = new_features

        returns[i_episode] = ep_return

    env.close()
    return returns


def train_torch_binary_agent(
    env_id: str,
    n_episodes: int,
    max_steps: int,
    n_features: int,
    n_closest: int,
    n_binary_features: int,
    n_binary_closest: int,
    alpha: float,
    epsilon: float,
    gamma: float,
    lmbda: float,
    seed: int,
    non_finite_bound: float,
) -> np.ndarray:
    env = gym.make(env_id)
    env.action_space.seed(seed)

    rep_space = _kanerva_observation_space(env, non_finite_bound)
    kanerva_layer = KanervaLayer(rep_space, n_features, n_closest)
    kanerva_binary_layer = KanervaBinary(n_features, n_binary_features, n_binary_closest)
    agent = QValueAgentTorchBinary(
        kanerva_layer=kanerva_layer,
        kanerva_binary_layer=kanerva_binary_layer,
        n_actions=env.action_space.n,
        alpha=alpha,
        epsilon=epsilon,
        gamma=gamma,
        lmbda=lmbda,
    )

    returns = np.zeros(n_episodes, dtype=np.float32)
    for i_episode in range(n_episodes):
        observation = _reset_env(env, seed=seed + i_episode)
        obs_tensor = torch.tensor(observation, dtype=torch.float32)

        ep_return = 0.0
        for _ in range(max_steps):
            action = agent.act(obs_tensor)
            new_observation, reward, done, _ = _step_env(env, action)
            new_obs_tensor = torch.tensor(new_observation, dtype=torch.float32)
            agent.update(obs_tensor, action, reward, new_obs_tensor)
            ep_return += reward

            if done:
                agent.erase_traces()
                break

            obs_tensor = new_obs_tensor

        returns[i_episode] = ep_return

    env.close()
    return returns


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """
    Computes a simple moving average using convolution.

    :param values: input 1D series
    :param window: smoothing window size
    :return: smoothed series (mode='same', so edge values are influenced by partial windows)
    """
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def _early_weighted_return(returns: np.ndarray) -> float:
    n_returns = len(returns)
    if n_returns == 0:
        return 0.0
    weights = _EARLY_RETURN_WEIGHTS_CACHE.get(n_returns)
    if weights is None:
        weights = np.linspace(EARLY_WEIGHT_START, EARLY_WEIGHT_END, n_returns)
        _EARLY_RETURN_WEIGHTS_CACHE[n_returns] = weights
    return float(np.average(returns, weights=weights))


def _tuning_objective_score(returns: np.ndarray, elapsed_seconds: float, speed_weight: float) -> float:
    """
    Objective units are reward points minus a speed penalty in seconds.
    Maximizing this favors strong early learning while penalizing slower trials.
    """
    return _early_weighted_return(returns) - speed_weight * elapsed_seconds


def _tune_pytorch_hyperparameters(args) -> tuple[dict, dict]:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna tuning requested, but optuna is not installed.") from exc

    n_closest_high = min(100, args.n_features - 1)
    n_binary_closest_high = min(100, args.n_binary_features - 1)

    def objective_torch(trial):
        n_closest = trial.suggest_int("n_closest", 5, n_closest_high)
        alpha = trial.suggest_float("alpha", 1e-3, 3e-1, log=True)
        epsilon = trial.suggest_float("epsilon", 0.01, 0.3)
        gamma = trial.suggest_float("gamma", 0.90, 0.999)
        lmbda = trial.suggest_float("lmbda", 0.80, 1.0)
        seed = (args.seed + trial.number * TORCH_SEED_MULTIPLIER) % SEED_MODULUS

        start = time.perf_counter()
        returns = train_torch_agent(
            env_id=args.env_id,
            n_episodes=args.optuna_episodes,
            max_steps=args.max_steps,
            n_features=args.n_features,
            n_closest=n_closest,
            alpha=alpha,
            epsilon=epsilon,
            gamma=gamma,
            lmbda=lmbda,
            seed=seed,
            non_finite_bound=args.non_finite_bound,
        )
        elapsed = time.perf_counter() - start
        return _tuning_objective_score(returns, elapsed, args.speed_weight)

    def objective_torch_binary(trial):
        n_closest = trial.suggest_int("n_closest", 5, n_closest_high)
        n_binary_closest = trial.suggest_int("n_binary_closest", 5, n_binary_closest_high)
        alpha = trial.suggest_float("alpha", 1e-3, 3e-1, log=True)
        epsilon = trial.suggest_float("epsilon", 0.01, 0.3)
        gamma = trial.suggest_float("gamma", 0.90, 0.999)
        lmbda = trial.suggest_float("lmbda", 0.80, 1.0)
        seed = (args.seed + trial.number * TORCH_BINARY_SEED_MULTIPLIER) % SEED_MODULUS

        start = time.perf_counter()
        returns = train_torch_binary_agent(
            env_id=args.env_id,
            n_episodes=args.optuna_episodes,
            max_steps=args.max_steps,
            n_features=args.n_features,
            n_closest=n_closest,
            n_binary_features=args.n_binary_features,
            n_binary_closest=n_binary_closest,
            alpha=alpha,
            epsilon=epsilon,
            gamma=gamma,
            lmbda=lmbda,
            seed=seed,
            non_finite_bound=args.non_finite_bound,
        )
        elapsed = time.perf_counter() - start
        return _tuning_objective_score(returns, elapsed, args.speed_weight)

    torch_sampler = optuna.samplers.TPESampler(seed=args.seed)
    torch_binary_sampler = optuna.samplers.TPESampler(seed=args.seed + TUNER_BINARY_SAMPLER_SEED_OFFSET)

    torch_study = optuna.create_study(direction="maximize", sampler=torch_sampler)
    torch_study.optimize(objective_torch, n_trials=args.optuna_trials)

    torch_binary_study = optuna.create_study(direction="maximize", sampler=torch_binary_sampler)
    torch_binary_study.optimize(objective_torch_binary, n_trials=args.optuna_trials)

    return torch_study.best_params, torch_binary_study.best_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NumPy and PyTorch Kanerva agent learning curves")
    parser.add_argument("--env-id", type=str, default="MountainCar-v0")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=2000)
    parser.add_argument("--n-closest", type=int, default=25)
    parser.add_argument("--n-binary-features", type=int, default=2000)
    parser.add_argument("--n-binary-closest", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smooth-window", type=int, default=10)
    parser.add_argument("--output", type=str, default="learning_curve_comparison.png")
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument(
        "--optuna-episodes",
        type=int,
        default=60,
        help="Episode budget per Optuna trial (kept lower than full training for faster tuning)",
    )
    parser.add_argument(
        "--speed-weight",
        type=float,
        default=1.0,
        help="Penalty multiplier for elapsed seconds during Optuna tuning objective",
    )
    parser.add_argument(
        "--non-finite-bound",
        type=float,
        default=5.0,
        help="Finite bound to replace infinite/near-infinite observation limits for Kanerva normalization",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch_params = {
        "n_closest": args.n_closest,
        "alpha": args.alpha,
        "epsilon": args.epsilon,
        "gamma": args.gamma,
        "lmbda": args.lmbda,
    }
    torch_binary_params = {
        "n_closest": args.n_closest,
        "n_binary_closest": args.n_binary_closest,
        "alpha": args.alpha,
        "epsilon": args.epsilon,
        "gamma": args.gamma,
        "lmbda": args.lmbda,
    }

    if args.optuna_trials > 0:
        if args.optuna_episodes <= 0:
            raise ValueError("--optuna-episodes must be > 0 when --optuna-trials is enabled")
        torch_best_params, torch_binary_best_params = _tune_pytorch_hyperparameters(args)
        torch_params.update(torch_best_params)
        torch_binary_params.update(torch_binary_best_params)

        print(f"Optuna best params (PyTorch): {torch_params}")
        print(f"Optuna best params (PyTorch+Binary): {torch_binary_params}")

    numpy_returns = train_numpy_agent(
        env_id=args.env_id,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        n_features=args.n_features,
        n_closest=args.n_closest,
        alpha=args.alpha,
        epsilon=args.epsilon,
        gamma=args.gamma,
        lmbda=args.lmbda,
        seed=args.seed,
        non_finite_bound=args.non_finite_bound,
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch_returns = train_torch_agent(
        env_id=args.env_id,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        n_features=args.n_features,
        n_closest=torch_params["n_closest"],
        alpha=torch_params["alpha"],
        epsilon=torch_params["epsilon"],
        gamma=torch_params["gamma"],
        lmbda=torch_params["lmbda"],
        seed=args.seed,
        non_finite_bound=args.non_finite_bound,
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch_binary_returns = train_torch_binary_agent(
        env_id=args.env_id,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        n_features=args.n_features,
        n_closest=torch_binary_params["n_closest"],
        n_binary_features=args.n_binary_features,
        n_binary_closest=torch_binary_params["n_binary_closest"],
        alpha=torch_binary_params["alpha"],
        epsilon=torch_binary_params["epsilon"],
        gamma=torch_binary_params["gamma"],
        lmbda=torch_binary_params["lmbda"],
        seed=args.seed,
        non_finite_bound=args.non_finite_bound,
    )

    episodes = np.arange(args.episodes)
    numpy_smooth = moving_average(numpy_returns, args.smooth_window)
    torch_smooth = moving_average(torch_returns, args.smooth_window)
    torch_binary_smooth = moving_average(torch_binary_returns, args.smooth_window)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, numpy_returns, color="tab:blue", alpha=0.25, label="NumPy (raw)")
    plt.plot(episodes, torch_returns, color="tab:orange", alpha=0.25, label="PyTorch (raw)")
    plt.plot(episodes, torch_binary_returns, color="tab:green", alpha=0.25, label="PyTorch+Binary (raw)")
    plt.plot(episodes, numpy_smooth, color="tab:blue", linewidth=2.0, label=f"NumPy (MA {args.smooth_window})")
    plt.plot(episodes, torch_smooth, color="tab:orange", linewidth=2.0, label=f"PyTorch (MA {args.smooth_window})")
    plt.plot(
        episodes,
        torch_binary_smooth,
        color="tab:green",
        linewidth=2.0,
        label=f"PyTorch+Binary (MA {args.smooth_window})",
    )
    plt.title(f"{args.env_id} Learning Curve: NumPy vs PyTorch vs PyTorch+Binary Agents")
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)

    print(f"Saved learning curve comparison to {args.output}")
    print(f"NumPy mean return:   {numpy_returns.mean():.2f}")
    print(f"PyTorch mean return: {torch_returns.mean():.2f}")
    print(f"PyTorch+Binary mean return: {torch_binary_returns.mean():.2f}")


if __name__ == "__main__":
    main()
