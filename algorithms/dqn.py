from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from env.toy_power_env import ToyPowerEnv


def train_dqn(total_timesteps=100_000, seed=0, verbose=1):
    env = ToyPowerEnv(seed=seed)
    env = Monitor(env)  # tracks per-episode reward/length for training logs

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=50_000,        # replay buffer size
        learning_starts=1_000,     # steps of random exploration before learning starts
        batch_size=64,
        gamma=0.99,                # discount factor - weight on future reward
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.3,   # fraction of training spent decaying epsilon
        exploration_final_eps=0.05,
        verbose=verbose,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps)
    return model
