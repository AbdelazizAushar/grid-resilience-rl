import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ToyPowerEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # ---- physical constants (kept simple/round for easy debugging) ----
    EPISODE_LENGTH = 24          # hours per episode
    BATTERY_CAPACITY = 10.0      # kWh, max battery can hold
    BATTERY_MAX_RATE = 2.0       # kWh, max charge/discharge per hour
    LOAD_BASE = 1.0              # kWh, base load demand
    LOAD_PEAK = 3.0              # kWh, peak-hour load demand
    OUTAGE_PROB_PEAK = 0.4       # probability grid is down during peak hours
    OUTAGE_PROB_OFFPEAK = 0.05   # probability grid is down off-peak
    PEAK_HOURS = set(range(17, 21))  # 5pm-9pm, evening peak

    # ---- reward weights (starting point - see step 7 discussion) ----
    PENALTY_UNMET_LOAD = -100.0   # per kWh of critical load unmet
    PENALTY_GRID_USE = -1.0       # per kWh charged from grid
    # per kWh shed (better than unmet, worse than nothing)
    PENALTY_SHED = -5.0
    # tried something impossible (e.g. discharge empty battery)
    PENALTY_INVALID_ACTION = -2.0

    # ---- actions ----
    ACTION_IDLE = 0
    ACTION_CHARGE_FROM_GRID = 1
    ACTION_DISCHARGE_TO_LOAD = 2
    ACTION_SHED_LOAD = 3
    ACTION_NAMES = ["idle", "charge_from_grid",
                    "discharge_to_load", "shed_load"]

    def __init__(self, seed=None):
        super().__init__()

        self.action_space = spaces.Discrete(4)

        # state: [hour_of_day (0-23), grid_status (0/1), battery_pct (0-1), load_demand (kWh)]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([23, 1, 1, self.LOAD_PEAK], dtype=np.float32),
            dtype=np.float32,
        )

        self._np_random = np.random.default_rng(seed)

        # internal state, set properly in reset()
        self.hour = 0
        self.battery_charge = 0.0  # kWh, actual stored amount (not %)
        self.grid_status = 1
        self.load_demand = self.LOAD_BASE

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self.hour = 0
        self.battery_charge = self.BATTERY_CAPACITY * 0.5  # start half-charged
        self.grid_status = self._roll_grid_status(self.hour)
        self.load_demand = self._roll_load_demand(self.hour)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = 0.0
        info = {"invalid_action": False, "unmet_load": 0.0}

        # ---- 1. WORLD UPDATE (must happen before applying agent's action -
        #         action consequences are computed relative to these values) ----
        # NOTE: grid_status and load_demand for THIS hour were already rolled
        # either in reset() or at the end of the previous step(), so the agent's
        # observation matches what it's about to act on. See _roll_* helpers.

        remaining_load = self.load_demand

        # ---- 2. APPLY AGENT'S ACTION (check -> clamp -> update) ----
        if action == self.ACTION_IDLE:
            # no resource action; load either met by grid (if on) or unmet
            pass

        elif action == self.ACTION_CHARGE_FROM_GRID:
            if self.grid_status == 1:
                room_in_battery = self.BATTERY_CAPACITY - self.battery_charge
                actual_charge = min(self.BATTERY_MAX_RATE, room_in_battery)
                self.battery_charge += actual_charge
                reward += self.PENALTY_GRID_USE * actual_charge
            else:
                # tried to charge from grid while grid is down - invalid/impossible
                reward += self.PENALTY_INVALID_ACTION
                info["invalid_action"] = True

        elif action == self.ACTION_DISCHARGE_TO_LOAD:
            if self.battery_charge <= 0:
                reward += self.PENALTY_INVALID_ACTION
                info["invalid_action"] = True
            else:
                actual_discharge = min(
                    self.BATTERY_MAX_RATE, self.battery_charge, remaining_load
                )
                self.battery_charge -= actual_discharge  # CLAMPED - never goes negative
                remaining_load -= actual_discharge

        elif action == self.ACTION_SHED_LOAD:
            shed_amount = remaining_load  # shed whatever's left this hour
            remaining_load = 0.0
            reward += self.PENALTY_SHED * shed_amount

        # ---- 3. HANDLE REMAINING LOAD (grid covers it if on and agent didn't
        #         already handle it; otherwise it's unmet) ----
        if remaining_load > 0:
            if action != self.ACTION_SHED_LOAD and self.grid_status == 1:
                # grid silently covers whatever's left, if available
                # (idle action relies on this - matches real life: if grid's on
                # and you did nothing else, grid just powers your load)
                remaining_load = 0.0
            else:
                info["unmet_load"] = remaining_load
                reward += self.PENALTY_UNMET_LOAD * remaining_load

        # ---- 4. UPDATE STATE for next step ----
        self.hour += 1
        terminated = False
        truncated = self.hour >= self.EPISODE_LENGTH

        if not truncated:
            self.grid_status = self._roll_grid_status(self.hour)
            self.load_demand = self._roll_load_demand(self.hour)

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def _roll_grid_status(self, hour):
        prob_out = self.OUTAGE_PROB_PEAK if hour in self.PEAK_HOURS else self.OUTAGE_PROB_OFFPEAK
        return 0 if self._np_random.random() < prob_out else 1

    def _roll_load_demand(self, hour):
        base = self.LOAD_PEAK if hour in self.PEAK_HOURS else self.LOAD_BASE
        noise = self._np_random.uniform(-0.1, 0.1)
        return max(0.0, base + noise)

    def _get_obs(self):
        battery_pct = self.battery_charge / self.BATTERY_CAPACITY
        return np.array(
            [self.hour, self.grid_status, battery_pct, self.load_demand],
            dtype=np.float32,
        )

    def render(self):
        status = "UP" if self.grid_status else "DOWN"
        print(
            f"hour={self.hour:2d} grid={status:4s} "
            f"battery={self.battery_charge:.2f}/{self.BATTERY_CAPACITY} "
            f"load={self.load_demand:.2f}"
        )
