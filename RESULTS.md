# Results Log — Grid Resilience RL

Tracks baseline and training results as the project progresses. Each entry should
record: what changed, what config/seed was used, and the resulting numbers —
so results are reproducible and regressions are easy to spot.

---

## Environment: ToyPowerEnv (v0 — stripped down)

Scope: grid + battery + load only (no solar/generator/outage-history yet),
24-hour episodes. See `env/toy_power_env.py`.

### Random Policy Baseline

Purpose: sanity-check the environment (not the algorithm). Confirms no physically impossible states occur and reward scale is sane,
before trusting any training results.

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

---

## Open items / things to verify

- [ ] Confirm why the two random-baseline runs above differ (-341 vs -393 mean) —
      check whether `env/toy_power_env.py` constants or `random_baseline.py` seeding
      changed between runs. Results should be exactly reproducible given identical
      code + seeds.
- [ ] Write heuristic baseline policy and log results.
- [ ] Wire up DQN (Stable-Baselines3) and compare against both baselines.
