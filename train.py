from policies.random_policy import random_action
from policies.heuristic_policy import heuristic_action
from evaluate import evaluate_policy


def run_random_baseline(num_episodes=100):
    return evaluate_policy(random_action, policy_name="Random Policy",
                           num_episodes=num_episodes)


def run_heuristic_baseline(num_episodes=100):
    return evaluate_policy(heuristic_action, policy_name="Heuristic Policy",
                           num_episodes=num_episodes)


if __name__ == "__main__":
    run_random_baseline()
    run_heuristic_baseline()
