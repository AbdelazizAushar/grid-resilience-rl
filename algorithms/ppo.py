from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from env.toy_power_env import ToyPowerEnv


def train_ppo(total_timesteps=100_000, seed=0, verbose=1):
    env = ToyPowerEnv(seed=seed)
    env = Monitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,   # SB3 default for PPO
        n_steps=256,           # steps collected per policy update (larger than A2C's n_steps=5)
        batch_size=64,
        n_epochs=10,            # number of passes over each collected batch
        gamma=0.99,             # discount factor - same reasoning as DQN/A2C
        gae_lambda=0.95,        # generalized advantage estimation smoothing
        clip_range=0.2,         # PPO's signature clipped objective - limits policy update size
        ent_coef=0.0,           # entropy bonus - encourages exploration via policy randomness
        vf_coef=0.5,            # weight on value-function loss vs policy loss
        verbose=verbose,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps)
    return model