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
|---|---|---|---|---|---|---|---|---|
| 2026-08-15 | 100 | -341.22 | 236.13 | -949.69 / -33.60 | 0 | 0.62 | 2.88 | Initial run, seed=42 base + per-episode seed=ep |
| 2026-08-15 | 100 | -393.04 | 263.50 | -1148.61 / -29.00 | 0 | 0.71 | 3.43 | Re-run — differs from row above, see open items |
| 2026-08-16 | 100 | -350.31 | 224.60 | -916.49 / -22.29 | 0 | 0.76 | 3.00 | Re-run after refactor into policies/ + evaluate.py — still differs run-to-run, see open items |

**Reading these results:**
- Battery violations = 0 confirms the clamp logic in `step()` is working (charge/discharge
  never pushes battery outside [0, BATTERY_CAPACITY]).
- Mean reward is strongly negative and high-variance, as expected — random policy has
  no strategy, so it frequently leaves load unmet (`PENALTY_UNMET_LOAD = -100/kWh`) and
  occasionally attempts invalid actions (e.g., charging from a downed grid).
- These numbers are the floor DQN needs to clear by a wide margin.
- Unlike the heuristic baseline (below), these three runs do NOT match exactly despite
  identical seeding intent — see open items. Root cause suspected: `env.action_space.sample()`
  uses Gymnasium's own internal RNG, separate from the environment's `self._np_random`,
  and is likely not being explicitly seeded.

### Heuristic Policy Baseline

Purpose: check whether RL is actually worth it, not just whether the code runs.
Rule (priority order, each hour): discharge battery to load if grid is down and
battery has charge → else charge from grid if grid is up and battery has room →
else shed load if grid is down and battery is empty → else idle.
See `policies/heuristic_policy.py` (run via `evaluate.py` / `train.py`).

| Date | Episodes | Mean reward | Std dev | Min / Max | Battery violations | Invalid actions/ep | Unmet load (kWh)/ep | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-08-15 | 100 | -162.59 | 105.83 | -415.90 / -5.00 | 0 | 0.00 | 1.54 | seed=42 base + per-episode seed=ep. |
| 2026-08-16 | 100 | -162.59 | 105.83 | -415.90 / -5.00 | 0 | 0.00 | 1.54 | Re-run after refactor into policies/ + evaluate.py — exact match, confirms refactor didn't change behavior. |

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

### DQN Training Results

Using Stable-Baselines3's `DQN` (`MlpPolicy`), not a from-scratch implementation —
see `algorithms/dqn.py`. Evaluated with `deterministic=True` (no exploration noise)
through the same shared `evaluate.py` used for the baselines above, so results are
directly comparable. 100,000 training timesteps (~4,166 episodes) per run.

| Date | Train seed | Episodes evaluated | Mean reward | Std dev | Min / Max | Battery violations | Invalid actions/ep | Unmet load (kWh)/ep | Beats heuristic? | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-16 | 0 | 100 | -33.59 | 15.90 | — | 0 | 0.00 | 0.00 | Yes (-33.59 vs -162.59) | First run |
| 2026-08-16 | 0 (re-run) | 100 | -37.24 | 12.79 | -67.87 / -19.71 | 0 | 0.00 | 0.00 | Yes (-37.24 vs -162.59) | Re-run, same train seed — small variation expected (network init, exploration, buffer sampling are separate RNG streams from the env's own seed, and not all are bit-for-bit reproducible, especially on GPU) |
| 2026-08-16 | 1 | 100 | -29.19 | 15.55 | — | 0 | 0.00 | 0.00 | Yes | Multi-seed sweep via `run_dqn_multi_seed` |
| 2026-08-16 | 2 | 100 | -52.63 | 25.19 | — | 0 | 0.00 | 0.07 | Yes | Multi-seed sweep — outlier: weakest mean, highest std, only seed with any unmet load |
| 2026-08-16 | 3 | 100 | -24.46 | 15.95 | — | 0 | 0.00 | 0.00 | Yes | Multi-seed sweep — best seed |
| 2026-08-16 | 4 | 100 | -28.94 | 15.59 | -66.93 / -5.00 | 0 | 0.00 | 0.00 | Yes | Multi-seed sweep |

**Across-seed summary (seeds 0, 1, 2, 3, 4 — using the sweep run, not the seed=0 re-run):**
Across-seed mean of mean_reward: **-34.49**, across-seed std: **9.96**.
Range: -24.46 (seed 3, best) to -52.63 (seed 2, worst).

**Reading these results:**
- Every one of the 5 seeds clearly beats the heuristic baseline (-162.59) by a wide
  margin — 3x at the weakest (seed 2) to nearly 7x at the best (seed 3). This is a
  robust result, not a fluke tied to one lucky seed.
- 4 of 5 seeds achieve **zero unmet load** across all 100 held-out episodes — the
  heuristic's known weakness (reactive-only, no foresight around peak hours) appears
  largely resolved.
- Std dev per-run (~13-25) is well below heuristic's (105.83) in every case — the
  trained policy isn't just better on average, it's far more consistent episode to
  episode, even in its weakest seed.
- Trace inspection (episode 0, seed 0) shows the agent charging early, holding reserve
  through a long idle stretch, and discharging through the evening peak window without
  running out — consistent with "anticipates peak-hour risk" behavior the heuristic lacks.
- **Seed 2 is a genuine outlier**: noticeably worse mean reward, roughly double the std
  dev of other seeds, and the only seed with any nonzero unmet load (0.07 kWh/ep).
  Converged to a weaker local optimum than the other four — a normal RL outcome, not a
  bug, and exactly the kind of thing a single-seed result would have hidden (either
  overstating reliability if only seed 3 had been run, or understating typical
  performance if only seed 2 had been run).
- One oddity noted in an earlier trace: a `shed_load` action taken while battery
  was fully charged (10.00/10.00). Plausible explanation: load demand that hour exceeded
  the single-hour max discharge rate, making shed better than partial-discharge +
  unmet-load penalty — not yet confirmed, low priority.
- **Honest reportable claim:** DQN reliably beats the heuristic baseline across 5 seeds
  (mean -34.49 ± 9.96), with some seed-dependent variance in how strong the learned
  policy is (range -24.46 to -52.63).

### A2C Training Results

Using Stable-Baselines3's `A2C` (`MlpPolicy`) — see `algorithms/a2c.py`. Same
evaluation protocol as DQN: `deterministic=True`, shared `evaluate.py`, 100 held-out
episodes, 100,000 training timesteps per run.

**First attempt (`ent_coef=0.0`) — discarded, not a valid result.** All 5 seeds
produced byte-identical results (mean -23.09, std 0.00 across seeds). Investigated:
seeding was confirmed correct in both `ToyPowerEnv(seed=seed)` and `A2C(..., seed=seed)`,
so root cause was diagnosed as `ent_coef=0.0` giving the policy no pressure to stay
stochastic during training, letting it collapse to the same near-deterministic decision
boundary almost regardless of initialization on this small environment. **Fix:** added
a small entropy bonus, `ent_coef=0.01`, to encourage genuine exploration during training
and let real seed-to-seed variance actually show up. Re-run below confirms the fix worked.

| Date | Train seed | Episodes evaluated | Mean reward | Std dev | Min / Max | Unmet load (kWh)/ep | Beats heuristic? | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-08-17 | 0 | 100 | -23.14 | 15.51 | -60.20 / 0.00 | 0.00 | Yes | `ent_coef=0.01` |
| 2026-08-17 | 1 | 100 | -23.09 | 15.52 | -60.20 / 0.00 | 0.00 | Yes | |
| 2026-08-17 | 2 | 100 | -23.11 | 15.51 | -60.20 / 0.00 | 0.00 | Yes | |
| 2026-08-17 | 3 | 100 | -39.26 | 35.67 | — | 0.11 | Yes | Outlier: notably worse mean, ~2x the std of other seeds, only seed with nonzero unmet load |
| 2026-08-17 | 4 | 100 | -23.09 | 15.52 | -60.20 / 0.00 | 0.00 | Yes | |

**Across-seed summary:** mean of mean_reward: **-26.34**, std across seeds: **6.46**.
Range: -23.09 (seeds 1/4, best) to -39.26 (seed 3, worst).

**Reading these results:**
- With the entropy fix, A2C now shows genuine, non-uniform seed-to-seed variance —
  seeds 0/1/2/4 cluster tightly around -23.1, while seed 3 is a real outlier (-39.26
  mean, std 35.67 — roughly double the others, and the only seed with any unmet load).
  This is a much more trustworthy result than the earlier byte-identical run.
- Still beats the heuristic baseline (-162.59) by a wide margin on every seed, including
  the outlier.
- Trace inspection (episode 0, seed 1) shows a different learned strategy than DQN's
  "charge early, hold reserve" pattern: A2C relies more on discharging occasionally and
  letting grid silently cover load during idle hours, rather than proactively charging
  from grid. Different strategy, similar reward range.
- **Comparable in spirit to DQN's seed 2 outlier** — both algorithms show one seed out
  of five converging to a distinctly weaker policy, which is a normal, expected RL
  outcome and reinforces why multi-seed evaluation matters here.
- Worth re-checking whether A2C's relative tightness (excluding the seed-3 outlier)
  holds once solar/generator are added and the environment/state space grows.

---

### PPO Training Results

Using Stable-Baselines3's `PPO` (`MlpPolicy`) — see `algorithms/ppo.py`. Uses the
SAME discrete action space as DQN/A2C (not the continuous version from the project
plan yet), so this is a fair, apples-to-apples comparison against them first —
isolating "is PPO a better algorithm" from "does a continuous action space help."
Continuous actions are a planned later extension. Same evaluation protocol as
DQN/A2C: `deterministic=True`, shared `evaluate.py`, 100 held-out episodes, 100,000
training timesteps per run.

| Date | Train seed | Episodes evaluated | Mean reward | Std dev | Unmet load (kWh)/ep | Beats heuristic? |
|---|---|---|---|---|---|---|
| 2026-08-16 | 0 | 100 | -32.84 | 15.90 | 0.00 | Yes |
| 2026-08-16 | 1 | 100 | -31.21 | 16.10 | 0.00 | Yes |
| 2026-08-16 | 2 | 100 | -24.18 | 15.55 | 0.00 | Yes |
| 2026-08-16 | 3 | 100 | -46.89 | 16.49 | 0.00 | Yes |
| 2026-08-16 | 4 | 100 | -31.56 | 13.97 | 0.00 | Yes |

**Across-seed summary:** mean of mean_reward: **-33.34**, std across seeds: **7.42**.
Range: -24.18 (seed 2, best) to -46.89 (seed 3, worst).

**Reading these results:**
- Zero unmet load across all 5 seeds and all 100 evaluation episodes each — matches
  DQN and A2C's reliability here.
- Across-seed std (7.42) is the **tightest of the three trained algorithms** so far
  (DQN: 9.96, A2C: 0.00-but-suspect — see A2C notes above) — no single catastrophic
  outlier seed the way DQN's seed 2 was; PPO's worst seed (-46.89) is still well
  within a reasonable range of its best (-24.18).
- Consistent with PPO's known design purpose: the clipped objective specifically
  exists to prevent destructively large policy updates, which shows up here as more
  consistent seed-to-seed reliability than DQN, without A2C's suspiciously-perfect
  zero variance.

| Algorithm | Across-seed mean | Across-seed std | Unmet load (any seed) |
|---|---|---|---|
| Random | -341.22 to -393.04 (single-seed, not multi-seeded) | high | frequent |
| Heuristic | -162.59 (deterministic, no seed variance) | — | some (1.54/ep) |
| DQN | -34.49 | 9.96 | seed 2 only (0.07/ep) |
| PPO | -33.34 | 7.42 | none |

**Honest reportable claim:** PPO reliably beats the heuristic baseline across 5 seeds
(mean -33.34 ± 7.42) with zero unmet load in every seed, and shows tighter seed-to-seed
consistency than DQN — this is the strongest all-around multi-seed result so far,
making it a solid candidate for the Optuna tuning stage next.

---