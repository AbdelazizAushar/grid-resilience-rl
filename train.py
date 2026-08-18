from policies.random_policy import random_action
from policies.heuristic_policy import heuristic_action
from policies.trained_policy import make_trained_action_fn
from algorithms.dqn import train_dqn
from algorithms.a2c import train_a2c
from algorithms.ppo import train_ppo
from evaluate import evaluate_policy
import numpy as np


def run_random_baseline(num_episodes=100):
    return evaluate_policy(random_action, policy_name="Random Policy",
                           num_episodes=num_episodes)


def run_heuristic_baseline(num_episodes=100):
    return evaluate_policy(heuristic_action, policy_name="Heuristic Policy",
                           num_episodes=num_episodes)


def run_dqn(total_timesteps=100_000, train_seed=0, num_episodes=100):
    print("Training DQN...")
    model = train_dqn(total_timesteps=total_timesteps, seed=train_seed)
    print("\nTraining complete. Evaluating...")
    action_fn = make_trained_action_fn(model)
    results = evaluate_policy(action_fn, policy_name="Trained DQN",
                              num_episodes=num_episodes)
    return model, results


def run_dqn_multi_seed(seeds=(0, 1, 2, 3, 4), total_timesteps=100_000, num_episodes=100):
    return _run_multi_seed(run_dqn, "DQN", seeds=seeds,
                           total_timesteps=total_timesteps, num_episodes=num_episodes)


def run_a2c(total_timesteps=100_000, train_seed=0, num_episodes=100):
    print("Training A2C...")
    model = train_a2c(total_timesteps=total_timesteps, seed=train_seed)
    print("\nTraining complete. Evaluating...")
    action_fn = make_trained_action_fn(model)
    results = evaluate_policy(action_fn, policy_name="Trained A2C",
                              num_episodes=num_episodes)
    return model, results


def run_a2c_multi_seed(seeds=(0, 1, 2, 3, 4), total_timesteps=100_000, num_episodes=100):
    return _run_multi_seed(run_a2c, "A2C", seeds=seeds,
                           total_timesteps=total_timesteps, num_episodes=num_episodes)


def run_ppo(total_timesteps=100_000, train_seed=0, num_episodes=100):
    print("Training PPO...")
    model = train_ppo(total_timesteps=total_timesteps, seed=train_seed)
    print("\nTraining complete. Evaluating...")
    action_fn = make_trained_action_fn(model)
    results = evaluate_policy(action_fn, policy_name="Trained PPO",
                              num_episodes=num_episodes)
    return model, results


def run_ppo_multi_seed(seeds=(0, 1, 2, 3, 4), total_timesteps=100_000, num_episodes=100):
    return _run_multi_seed(run_ppo, "PPO", seeds=seeds,
                           total_timesteps=total_timesteps, num_episodes=num_episodes)


def _run_multi_seed(run_fn, algo_name, seeds, total_timesteps, num_episodes):
    all_results = []

    for seed in seeds:
        print(f"\n{'=' * 50}")
        print(f"{algo_name} - training seed {seed}")
        print(f"{'=' * 50}")
        model, results = run_fn(total_timesteps=total_timesteps,
                                train_seed=seed, num_episodes=num_episodes)
        results["train_seed"] = seed
        all_results.append(results)

    mean_rewards = [r["mean_reward"] for r in all_results]
    across_seed_mean = float(np.mean(mean_rewards))
    across_seed_std = float(np.std(mean_rewards))

    print(f"\n{'=' * 50}")
    print(f"{algo_name} — summary across seeds")
    print(f"{'=' * 50}")
    for r in all_results:
        print(f"  seed={r['train_seed']}: mean_reward={r['mean_reward']:.2f}, "
              f"std_reward={r['std_reward']:.2f}, "
              f"unmet_load={r['mean_unmet_load']:.2f}")
    print(f"\nAcross-seed mean of mean_reward: {across_seed_mean:.2f}")
    print(f"Across-seed std of mean_reward:  {across_seed_std:.2f}")

    return all_results


if __name__ == "__main__":
    # run_random_baseline()
    # run_heuristic_baseline()
    # run_dqn()
    # run_a2c()
    run_a2c_multi_seed()
    # run_ppo_multi_seed()
