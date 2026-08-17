from policies.random_policy import random_action
from policies.heuristic_policy import heuristic_action
from policies.trained_policy import make_trained_action_fn
from algorithms.dqn import train_dqn
from evaluate import evaluate_policy


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


if __name__ == "__main__":
    run_random_baseline()
    run_heuristic_baseline()
    run_dqn()
