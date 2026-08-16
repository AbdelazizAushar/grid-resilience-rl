import numpy as np

from env.toy_power_env import ToyPowerEnv


def evaluate_policy(policy_fn, policy_name="policy", num_episodes=100,
                    verbose_first_episode=True, base_seed=42):
    env = ToyPowerEnv(seed=base_seed)
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
            print(f"=== Episode 0 trace ({policy_name}) ===")

        while not (terminated or truncated):
            action = policy_fn(env, obs)
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

    results = {
        "policy_name": policy_name,
        "num_episodes": num_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "battery_violations": battery_violations,
        "mean_invalid_actions": float(np.mean(invalid_action_counts)),
        "mean_unmet_load": float(np.mean(unmet_load_totals)),
        "episode_rewards": episode_rewards,
    }

    print(f"\n=== {policy_name} Evaluation Results ===")
    print(f"Episodes run:              {results['num_episodes']}")
    print(f"Mean episode reward:       {results['mean_reward']:.2f}")
    print(f"Std dev episode reward:    {results['std_reward']:.2f}")
    print(
        f"Min / Max episode reward:  {results['min_reward']:.2f} / {results['max_reward']:.2f}")
    print(
        f"Battery constraint violations: {results['battery_violations']} (should be 0)")
    print(
        f"Mean invalid actions/episode:  {results['mean_invalid_actions']:.2f}")
    print(f"Mean unmet load (kWh)/episode: {results['mean_unmet_load']:.2f}")

    return results
