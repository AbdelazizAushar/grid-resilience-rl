# Results Log — Grid Resilience RL

Tracks baseline and training results as the project progresses. Each entry should
record: what changed, what config/seed was used, and the resulting numbers —
so results are reproducible and regressions are easy to spot.

---

## Environment: ToyPowerEnv (v0 — stripped down)

Scope: grid + battery + load only (no solar/generator/outage-history yet),
24-hour episodes. See `env/toy_power_env.py`.

### Random Policy Baseline

Purpose: sanity-check the environment (not the algorithm). Confirms no physically
impossible states occur and reward scale is sane, before trusting any training results.

| Date | Episodes | Mean reward | Std dev | Min / Max | Battery violations | Invalid actions/ep | Unmet load (kWh)/ep | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-15 | 100 | -341.22 | 236.13 | -949.69 / -33.60 | 0 | 0.62 | 2.88 | Initial run, seed=42 base + per-episode seed=ep |
| 2026-08-15 | 100 | -393.04 | 263.50 | -1148.61 / -29.00 | 0 | 0.71 | 3.43 | Re-run — TODO: confirm whether config/constants changed vs. row above, since numbers should be deterministic given same code+seeds |

**Reading these results:**

- Battery violations = 0 confirms the clamp logic in `step()` is working (charge/discharge
  never pushes battery outside [0, BATTERY_CAPACITY]).
- Mean reward is strongly negative and high-variance, as expected — random policy has
  no strategy, so it frequently leaves load unmet (`PENALTY_UNMET_LOAD = -100/kWh`) and
  occasionally attempts invalid actions (e.g., charging from a downed grid).
- These numbers are the floor DQN needs to clear by a wide margin.

### Heuristic Policy Baseline

Purpose: check whether RL is actually worth it, not just whether the code runs.
Rule (priority order, each hour): discharge battery to load if grid is down and
battery has charge → else charge from grid if grid is up and battery has room →
else shed load if grid is down and battery is empty → else idle.
See `heuristic_baseline.py`.

| Date | Episodes | Mean reward | Std dev | Min / Max | Battery violations | Invalid actions/ep | Unmet load (kWh)/ep | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-15 | 100 | -162.59 | 105.83 | -415.90 / -5.00 | 0 | 0.00 | 1.54 | seed=42 base + per-episode seed=ep. Reproduced identically on re-run. |

**Reading these results:**

- Roughly halves mean reward penalty vs. random (-162.59 vs -341.22) and cuts unmet
  load nearly in half (1.54 vs 2.88 kWh/ep).
- Invalid actions = 0, as expected — the heuristic explicitly checks grid/battery
  status before acting, so it never attempts something impossible.
- Std dev (105.83) is still meaningful — day-to-day difficulty varies a lot depending
  on how outages line up with peak hours.
- Known weakness: the heuristic is purely reactive — once battery is full and grid is
  up, it idles indefinitely rather than anticipating the evening peak-outage window.
  Remaining unmet load (1.54 kWh/ep) comes from days where an outage during/near peak
  hours outlasts whatever charge was on hand. This is the specific gap a learned
  policy has room to close.
- This is the bar DQN needs to clear by a meaningful margin, not just barely beat.
