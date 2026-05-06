# Diagnostic Session Summary — 2026-05-05/06

End-of-session record before report writing. Captures the four headline
landings, the belief-vs-truth analysis on Run 1, the launch-reliability
finding, and tomorrow's first command.

---

## Tomorrow's first command (paste into Claude)

> Don't change any launch files, code, world, model, or submodule.
> Infrastructure is frozen. Retarget the six analysis scripts
> (compare_nav_runs, analyze_flight_dynamics, analyze_map, plot_landmarks,
> plot_ekf_vs_odom_maps, quantify_slam_error) at
> `analysis/headline_bags/final_ekf_bag_run2/` and
> `analysis/headline_bags/final_odom_bag_run2/`. Save outputs to
> `results/figures/`. One script at a time, with verification gates. Start
> with compare_nav_runs.py.

The methodologically clean comparison is **run2 vs run2** because
`/crazyflie/range/down` was only added to `BAG_TOPICS` on the launches that
produced the run2 bags.

---

## Headline dataset — four bags, two-sample design

All bags safety-copied to `analysis/headline_bags/` and verified.

| Run | Pipeline | Landing distance from origin | `/crazyflie/range/down` recorded? |
|-----|----------|------------------------------|------------------------------------|
| 1   | EKF-SLAM | **4.2 cm**                  | no                                 |
| 1   | Odom + Brownian drift | **7.2 cm**     | no                                 |
| 2   | EKF-SLAM | **2.6 cm**                  | **yes**                            |
| 2   | Odom + Brownian drift | **5.4 cm**     | **yes**                            |

The "landing distance" reported here is **EKF-belief vs goal** — the orchestrator's
view, derived from `/ekf_pose`, which is what `mission_orchestrator.py` logs at
RETURN-leg landing. This is the natural framing for "did the controller hit
its target?" but it is not the only valid framing — see the next section.

The Run 2 EKF launch was the second attempt; the first stalled at the
planner stage (see Launch reliability section below).

---

## Belief-vs-truth analysis (Run 1 EKF, the only run with full diagnostic)

`/ekf_pose` (EKF's own belief) and `/crazyflie/odom` (Gazebo physics ground
truth, untouched by the bridge) diverge meaningfully over the 199 s mission.
Four equally-valid framings:

| Framing | Definition | EKF Run 1 | Odom Run 1 |
|---------|------------|-----------|------------|
| **A. Belief vs goal** | `/ekf_pose` final position vs (0, 0) — what the controller saw | 4.2 cm | 7.2 cm (~confirmed by truth) |
| **B. Truth vs goal** | `/crazyflie/odom` final position vs (0, 0) — physical landing accuracy | **9.96 cm** | 7.2 cm |
| **C. Belief minus truth at landing** | `\|/ekf_pose − /crazyflie/odom\|` at the last sample — pose-estimate error | 10.6 cm | 3.5 cm |
| **D. Drift rate** | C divided by mission duration (s) | 0.053 cm/s | ~0.075 cm/s |

What this means in plain language:

- The EKF *thought* it landed at (0.028, −0.024), 4.2 cm from origin. Physically,
  it landed at (0.013, +0.099), 9.96 cm from origin. The EKF's xy belief had
  drifted ~10 cm by mission end.
- The odom run is shorter (47 s) and its `/ekf_pose` is `/crazyflie/odom +
  injected Brownian drift`. Cumulative drift = 3.5 cm over that span.
- Comparing 199 s EKF vs 47 s odom is not apples-to-apples on **drift**
  metrics. It IS apples-to-apples on the **mission outcome** ("did the
  controller hit its target?") because both ran the same nav/planner stack
  with their respective pose source.

**Likely cause of the EKF drift:** 4-beam multiranger gives limited
observational diversity. Scan-to-map ICP is under-constrained along local
symmetry directions of the obstacle field; the down-range sensor pins z but
not xy; over 200 s the EKF's xy belief slides without internal contradiction.

**For the report:** any of A/B/C/D is defensible; pick whichever serves
the argument and disclose the others. Do not claim "EKF beats odom in
absolute landing accuracy" without qualifying — that claim is true under
framing A but false under framing B in Run 1. Run 2 may differ; rerun the
analysis on Run 2 before committing to a framing.

---

## Launch reliability finding

Run 2 EKF launch succeeded on the **second** attempt. The first attempt
stalled at the planner stage. Diagnostic on the failed bag (now at
`obsolete/results/final_ekf_bag_failed_*`):

| Signal              | Working Run 1 | Failed attempt |
|---------------------|---------------|----------------|
| `/planned_path` msgs | 384          | **11**         |
| `/cmd_vel` msgs      | 1973         | 710            |
| Trajectory extent x  | reaches 0.91 | stalls at 0.34 |
| Bag duration         | 199 s        | 64 s           |

The planner emitted **one initial 18-pose path at t=7.36s, one update at
t=12.35s, then nothing**. EKF, scans, mapper, odom were all flowing
normally — only the planner stopped issuing path updates. Drone stalled
near (0.13, −0.01), reversed to (0.06, −0.03), drifted south, and landed
prematurely.

**Note on `/goal_pose` counts in bags:** the launch's `bag_record` and
`publish_goal` are independent ExecuteProcess actions. Depending on which
subscribes first, `/goal_pose` may or may not be captured in the bag even
when it WAS published and the planner DID receive it. Run 1 caught it
(count=1). Run 2 missed it (count=0) but the planner clearly got it
(308 planned_path messages, mission completed). Treat `/goal_pose` count
as an unreliable bag-side signal — `/planned_path` count is the better
indicator of whether the planner had a target.

**Cause of the stall — UNIDENTIFIED.** Three candidates, in order of
plausibility:

1. **Goal-pose latching:** D* Lite needs the goal persistently. If the
   planner's internal goal latch was reset or `/goal_pose` was
   re-published with empty contents mid-flight, replanning would stop.
2. **Bag-record / publish-goal timing race:** see paragraph above. Less
   plausible as a *failure* cause since this is a recording artifact, not
   a runtime artifact.
3. **`mission_orchestrator.py` interference:** the orchestrator can
   publish `/goal_pose` mid-flight on the return leg. If it triggered
   prematurely or wrote a malformed goal, the planner could chase a
   phantom goal.

We did not isolate the cause. Recovered cleanly on retry. **Reliability
characterization for the report:** of 3 EKF launch attempts tonight that
got past hover, 2/3 produced a clean run-to-completion. Sample is small;
do not over-claim system reliability.

---

## Working directory state at session end

- **Branch:** `analysis-and-figures`, tracking `origin/analysis-and-figures`
- **Tag:** `headline-data-captured` pinned to pre-Run-2 commit (still valid as
  a fallback point — Run 1 dataset is reachable from there)
- **Safety copies:**
  - `analysis/headline_bags/final_ekf_bag_run1/`
  - `analysis/headline_bags/final_odom_bag_run1/`
  - `analysis/headline_bags/final_ekf_bag_run2/`
  - `analysis/headline_bags/final_odom_bag_run2/`
- **Live working copies:** `results/final_ekf_bag/`, `results/final_odom_bag/`
  (these can be wiped by the next launch's `clean_bag_dir` action; the
  safety copies are the durable record)
- **Failed/superseded attempts:** all under `obsolete/results/` with clear
  suffixes (see Obsolete-directory audit section in this commit)
- **`results/figures/`:** clean (empty); ready for retargeted-script output
- **Workspace integrity:** no runtime code/launch/world/model edits since the
  Run 1 commits. Submodule dirty as expected (down_range sensor, world
  obstacles); model.sdf has down_range gpu_lidar at line 74; mesh URIs
  relative; no absolute paths. Verified earlier in session.

---

## Don't do tomorrow

- Don't edit any launch file, runtime code, world SDF, model SDF, or the
  submodule. Infrastructure is frozen.
- Don't re-fly. The four headline bags are sufficient.
- Don't claim "EKF beats odom" without picking and disclosing a framing.
- Don't bundle cleanup + retargeting + new scripts into one Claude turn.
  Use one verification gate per script.
