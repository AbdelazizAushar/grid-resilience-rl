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
