from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

from env.toy_power_env import ToyPowerEnv


def train_a2c(total_timesteps=100_000, seed=0, verbose=1):
    env = ToyPowerEnv(seed=seed)
    env = Monitor(env)

    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=7e-4,   # SB3 default for A2C
        n_steps=5,             # steps collected per update (on-policy - no replay buffer)
        gamma=0.99,             # discount factor - same reasoning as DQN
        gae_lambda=1.0,         # generalized advantage estimation smoothing
        ent_coef=0.0,           # entropy bonus - encourages exploration via policy randomness
        vf_coef=0.5,            # weight on value-function loss vs policy loss
        verbose=verbose,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps)
    return model