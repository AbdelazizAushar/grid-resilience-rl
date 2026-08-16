def make_trained_action_fn(model):
    def trained_action(env, obs):
        action, _states = model.predict(obs, deterministic=True)
        return int(action)

    return trained_action
