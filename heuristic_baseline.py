import numpy as np
from env.toy_power_env import ToyPowerEnv


def heuristic_action(env, obs):
    hour, grid_status, battery_pct, load_demand = obs
    battery_charge = battery_pct * env.BATTERY_CAPACITY

    grid_up = grid_status > 0.5
    battery_has_charge = battery_charge > 1e-6
    battery_has_room = battery_charge < env.BATTERY_CAPACITY - 1e-6

    if not grid_up and battery_has_charge:
        return env.ACTION_DISCHARGE_TO_LOAD
    elif grid_up and battery_has_room:
        return env.ACTION_CHARGE_FROM_GRID
    elif not grid_up and not battery_has_charge:
        return env.ACTION_SHED_LOAD
    else:
        return env.ACTION_IDLE


def run_heuristic_policy(num_episodes=100, verbose_first_episode=True):
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
            print("=== Episode 0 trace (heuristic policy) ===")

        while not (terminated or truncated):
            action = heuristic_action(env, obs)
            obs, reward, terminated, truncated, step_info = env.step(action)

            total_reward += reward
            if step_info["invalid_action"]:
                invalid_actions += 1
            unmet_load += step_info["unmet_load"]

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

    print("\n=== Heuristic Policy Baseline Results ===")
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
    run_heuristic_policy(num_episodes=100)
