import optuna

from algorithms.ppo import train_ppo
from policies.trained_policy import make_trained_action_fn
from evaluate import evaluate_policy


def objective(trial, trial_timesteps=30_000, eval_episodes=30, train_seed=0):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    n_steps = trial.suggest_categorical("n_steps", [64, 128, 256, 512])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_epochs = trial.suggest_int("n_epochs", 3, 20)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.02)

    if n_steps % batch_size != 0:
        raise optuna.exceptions.TrialPruned()

    model = train_ppo(
        total_timesteps=trial_timesteps,
        seed=train_seed,
        verbose=0,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
    )

    action_fn = make_trained_action_fn(model)
    results = evaluate_policy(action_fn, policy_name=f"PPO trial {trial.number}",
                              num_episodes=eval_episodes,
                              verbose_first_episode=False)

    return results["mean_reward"]


def run_optuna_tuning(n_trials=30, trial_timesteps=30_000, eval_episodes=30,
                      train_seed=0):
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, trial_timesteps=trial_timesteps,
                                eval_episodes=eval_episodes, train_seed=train_seed),
        n_trials=n_trials,
    )

    print("\n" + "=" * 50)
    print("Optuna PPO tuning — best trial")
    print("=" * 50)
    print(f"Best mean_reward: {study.best_value:.2f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return study


if __name__ == "__main__":
    run_optuna_tuning()
