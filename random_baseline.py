import numpy as np
from env.toy_power_env import ToyPowerEnv


def run_random_policy(num_episodes=100, verbose_first_episode=True):
    env = ToyPowerEnv(seed=42)
    episode_rewards = []
    battery_violations = 0
    invalid_action_counts = []
    unmet_load_totals = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        total_reward = 0.0
        invalid_actions = 0
        unmet_load = 0.0
        terminated = truncated = False

        if verbose_first_episode and ep == 0:
            print("=== Episode 0 trace ===")

        while not (terminated or truncated):
            action = env.action_space.sample()  # pure random
            obs, reward, terminated, truncated, step_info = env.step(action)

            total_reward += reward
            if step_info["invalid_action"]:
                invalid_actions += 1
            unmet_load += step_info["unmet_load"]

            # sanity check: battery should NEVER be negative or over capacity
            battery_charge = obs[2] * env.BATTERY_CAPACITY
            if battery_charge < -1e-6 or battery_charge > env.BATTERY_CAPACITY + 1e-6:
                battery_violations += 1
                print(f"  BATTERY VIOLATION at ep {ep}: {battery_charge:.4f}")

            if verbose_first_episode and ep == 0:
                action_name = env.ACTION_NAMES[action]
                print(f"  action={action_name:18s} reward={reward:7.2f} "
                      f"battery={battery_charge:.2f}")

        episode_rewards.append(total_reward)
        invalid_action_counts.append(invalid_actions)
        unmet_load_totals.append(unmet_load)

    print("\n=== Random Policy Baseline Results ===")
    print(f"Episodes run:              {num_episodes}")
    print(f"Mean episode reward:       {np.mean(episode_rewards):.2f}")
    print(f"Std dev episode reward:    {np.std(episode_rewards):.2f}")
    print(
        f"Min / Max episode reward:  {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")
    print(f"Battery constraint violations: {battery_violations} (should be 0)")
    print(
        f"Mean invalid actions/episode:  {np.mean(invalid_action_counts):.2f}")
    print(f"Mean unmet load (kWh)/episode: {np.mean(unmet_load_totals):.2f}")

    return episode_rewards


if __name__ == "__main__":
    run_random_policy(num_episodes=100)
