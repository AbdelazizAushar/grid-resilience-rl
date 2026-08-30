from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from env.toy_power_env import ToyPowerEnv


def train_ppo(total_timesteps=100_000, seed=0, verbose=1,
              learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=10,
              gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
              vf_coef=0.5):
    env = ToyPowerEnv(seed=seed)
    env = Monitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=verbose,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps)
    return model
