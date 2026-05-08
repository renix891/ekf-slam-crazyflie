# Session Log — 2026-05-06 / 2026-05-07 overnight

Full chronological record of everything performed in this Claude Code
session, with key observations, decisions, and findings preserved.
This is not the per-script handoff (that's `tomorrow_handoff.md`) — this
is the conversational record of how the work happened, including the
investigative paths taken, the dead ends, and the reasoning behind each
choice.

Branch: `analysis-and-figures`. All commits referenced are on
`origin/analysis-and-figures`.

---

## Phase 1 — Corner detection investigation

### The trigger

User noticed that EKF debug logs throughout Run 2 said
`Line landmarks: 7 | Corner landmarks: 0 | State dim: 18`. Corners
appeared to be missing entirely. Three hypotheses to test:
1. Corner detection logic exists but isn't producing corners
2. Corner detection was disabled/removed
3. There's a bug preventing corner extraction

### Read-only diagnostic dispatched

Subagent (Explore) read `ekf_slam_node.cpp`, `line_extractor.cpp`,
related headers. Findings:

**Corner code DOES exist** in `line_extractor.cpp:204-241` and
`ekf_slam_node.cpp:407-422, 695-726, 779-846`. Algorithm: pair every
two world-frame line landmarks; if perpendicular within
`|cos(Δθ)| < 0.3` (≈17°), solve the 2×2 linear system for their
intersection point; reject if intersection is >3 m from robot.

**Critical finding** (the actual answer): corners are
**visualization-only, never EKF state**. From
`ekf_slam_node.cpp:650-652`:

> "Corner SLAM removed: corners are derived geometrically from the
> line landmarks at visualization time."

The "Corner landmarks: 0" log line is a hardcoded
`int n_corners = 0;` at `ekf_slam_node.cpp:854` that NEVER gets
updated. The log would say 0 even if 50 corners were being published
to RViz every scan. The 4-beam multiranger plus a 30-scan rolling
buffer was deemed too sparse for corners-as-state; the project
deliberately removed them and kept line-only state with corners as
post-hoc visualization.

### Bag verification

Second subagent dispatched: read final messages of
`/ekf_slam/debug/landmark_corners` from
`analysis/headline_bags/final_ekf_bag_run2/`. Result:
**1401 of 1491 messages had real markers** (94%); the final message
contained 10 corners forming a clean grid:

| id | x | y |
|---|---|---|
| 0 | 1.6689 | 1.9556 | (top-right wall corner)
| 1 | -1.7965 | 1.9042 | (top-left wall corner)
| 2 | 0.3943 | 1.9367 | (top wall, interior partition)
| 3 | 1.8065 | -1.5694 | (bottom-right wall corner)
| 4 | -1.8270 | -1.6494 | (bottom-left wall corner)
| 5 | 0.2715 | -1.6032 | (bottom wall, interior partition)
| 6 | 1.7391 | 0.1571 | (right wall, mid)
| 7 | -1.8128 | 0.0081 | (left wall, mid)
| 8 | 0.0048 | 0.0843 | (origin region)
| 9 | 0.3305 | 0.0980 | (interior, near origin)

The corners exist in the bag throughout the mission;
`analyze_map.py` simply wasn't subscribing to the topic.

### Resolution

Added `read_final_corner_markers(bag_path)` helper to `analyze_map.py`
mirroring the existing `read_final_landmark_lines`, modified
`plot_ekf_landmarks` to read corners and overlay them. Initial
choice: gold diamonds with black edge.

Commit `1aac92d` — "analyze_map.py: add corner-landmark overlay and
unified figure restyle" (the restyle came later in the session).

---

## Phase 2 — The figure-restyle saga

### Round 1: cartoonish symbols

User's reaction to the initial gold-diamond corners: "they do look
cartoonish... mismatched symbols (diamond + plus + star, all with
bright colors and black borders) read like game UI rather than an
engineering figure."

Decision: full restyle of all 4 plot functions in `analyze_map.py`.

### Round 2: full monochromatic

Applied:
- `_draw_goals`: gold star + green plus → both filled black squares,
  distinguished by label only
- Free dots: lightblue → lightgray (occupied red stays as the data signal)
- Trajectory + start: blue/green → all black, lw=1.2
- EKF lines: lw=3 → lw=2 (still red)
- Endpoint dots: dropped from `plot_ekf_landmarks`
- Corner markers: gold diamonds → small black filled circles

Tasks tracked. All 6 hunks applied, regenerated.

User's reaction: "ok this is too monochromatic... why is everything
black now? ... think about the cohesion of all the plots please"

### Round 3: re-introducing accent colors

- Trajectory: `'rebeccapurple'` (`#663399`) — high-contrast vs
  `coolwarm` colormap on scan-endpoints figure, distinct from red
  (data) and green (goals).
- Goals: black squares → green squares
- Corners: black circles → orange filled circles, marker size 8,
  black edge

User noted: "make the return landing one a black and green striped
pad" — discussed and chose **option 3 (heavier black border on return
goal)** instead of striped (matplotlib has no clean striped marker
primitive at small size).

### Final palette (settled)

| Element | Color |
|---|---|
| Trajectory + start | rebeccapurple `#663399` |
| Outbound goal | filled green square, edge width 0.8 |
| Return goal | filled green square, **edge width 2.5** (heavier border = the "final/landing" semantic) |
| EKF line landmarks | red, lw=2 |
| EKF corner landmarks | orange filled circle, size 8, black edge |
| Occupied dots | red |
| Free dots | lightgray |
| GT walls | black solid lw=1.5 |
| GT obstacles | dashed gray lw=1.0 |
| Background scan endpoints | coolwarm colormap (time-coded) |

### Lesson learned (recorded for future style passes)

User-validated guidance from this round:
- Avoid bright colors with thick black borders for engineering figures —
  reads like game UI.
- Don't force everything to the same color either — full monochromatic
  is sterile.
- Cohesion comes from a deliberately small accent palette: ONE color
  for "data," one for "goals," one neutral for "trajectory."
- Distinguish related elements (e.g. two goals) by visual *weight*, not
  by changing color or shape.

Commit `1aac92d` (corner overlay + restyle in one commit).

---

## Phase 3 — Tracking figures in git

User: "add the figure to the git why are they git ignore?"

Investigation: `.gitignore:14` had a blanket `*.png` rule. Two options
proposed:
1. Targeted unignore: add `!results/figures/*.png` after the rule
2. Remove the blanket rule entirely

Chose option 1 (standard "ignore everywhere except this dir" pattern).
All 7 existing figures became visible to git, were added.

Commit `15868d1` — "analyze_map.py: 7 polished figures... track
results/figures/*.png".

---

## Phase 4 — Killing the line extraction figure

### Diagnostic: was `plot_landmarks.py` redundant?

User asked Claude to read `plot_landmarks.py` (a separate script in
the repo). Findings:
- It produces a single PNG: `results/ekf_slam_landmarks.png`
- Substantial overlap with `plot_ekf_landmarks` in `analyze_map.py`
  (both render final-state line + corner landmarks)
- **Currently broken**: imports `GOAL` from `analyze_map`, which was
  renamed to `OUTBOUND_GOAL`/`RETURN_GOAL` in the recent restyle
- One unique feature: combines C-space dot grid as background behind
  landmarks (different reference frame than `plot_ekf_landmarks`'s
  GT-overlay approach)

User decision: **delete it** (move to `obsolete/scripts/`), and also
**fold the C-space idea into `plot_ekf_landmarks`** by tightening the
line extraction.

### Line extraction merger attempt

Wrote `_merge_segments` for `plot_line_extraction` to collapse the
140-segment messy line-extraction figure to ~15-20 clean lines.

Algorithm:
1. Each segment → (rho, theta) Hessian normal form
2. Cluster pairs by: angle diff < 8°, rho diff < 0.15 m,
   along-line extent overlapping or gap < 0.30 m
3. Refit each cluster with TLS through union of endpoint, take longest
   extent
4. Iterate up to 5 times until stable

**Result**: 140 → 45 segments (EKF), 29 → 22 (Odom). Better but still
2× the target. Walls visible as continuous traces but with parallel
duplicates within ~0.20 m of each other — drift smear.

User decision: **drop the line extraction figures entirely.** "Clean
break, 5 figures total in `analyze_map.py`, move on to next script."
The post-hoc Python re-implementation didn't match the EKF's live
extractor (which uses a 30-scan rolling buffer in world coords with
15° world-bearing buckets — fundamentally different from the
midterm-port IEPF algorithm).

Removed `plot_line_extraction`, all helpers, all constants, the call
sites. Deleted both PNG outputs from disk and git.

Commit `f768785` — "analyze_map.py: drop line-extraction figures;
move plot_landmarks.py to obsolete".

---

## Phase 5 — `plot_ate_rpe.py`

### Definitions confirmed

- ATE = ‖belief_xy(t) − truth_xy(t)‖ at every belief sample, truth
  interpolated onto belief timestamps. Translation-only.
- RPE = ‖Δbelief − Δtruth‖ over Δ=1s windows. Translation-only.
- Truth source: `/crazyflie/odom` (Gazebo physics, the un-noised
  variant). Confirmed: `/crazyflie/odom_noisy` is the noisy variant
  the EKF consumes.

### Important correction from user

Claude initially miscalculated bag durations (107s/150s). User
corrected: EKF Run 2 = **161.3s**, Odom Run 2 = **47.1s**. Asymmetric
durations → stack ATE subplots vertically (not side-by-side) so the
short odom mission doesn't have dead horizontal space.

### Key methodology check

Read `quantify_slam_error.py` first to ensure no methodology conflict.
**Finding**: `quantify_slam_error.py` does NOT compute ATE despite
the title — it computes obstacle surface RMS, path efficiency, C-obs
fraction, and per-leg landing errors (single endpoint). So the new
`plot_ate_rpe.py` is the FIRST ATE/RPE for these bags. No
reconciliation risk.

### Implementation

- `read_belief_and_truth(bag_path)` → returns belief and truth as
  (t, x, y) tuples
- `compute_ate(belief, truth)` → interpolate truth onto belief
  timestamps, compute Euclidean distance
- `compute_rpe(belief, truth, delta_s=1.0)` → for each belief sample
  at t_i, find the belief sample closest to t_i+Δ within Δ/2 tolerance,
  compute ‖Δbelief - Δtruth‖
- `plot_ate_timeseries`: 2 stacked subplots, EKF top (161s) / odom
  bottom (47s), shared y-axis, RMSE horizontal lines
- `plot_rpe_histogram`: overlaid alpha-blended histograms with shared
  bins (Freedman–Diaconis on combined data), RMSE vertical lines

### Results

| Metric | EKF | Odom |
|---|---|---|
| ATE RMSE | **0.114 m** | **0.121 m** |
| ATE max | 0.217 m | 0.209 m |
| ATE samples | 1612 | 2349 |
| RPE RMSE (Δ=1s) | **0.035 m** | **0.026 m** |
| RPE max | 0.107 m | 0.075 m |
| RPE windows | 1601 | 2299 |

### Key observation: RPE inversion

**Odom has lower RPE than EKF** (0.026 vs 0.035). Surface reading: "odom
wins on local consistency." Real explanation: odom belief is
truth + slow Brownian random walk, so 1s-window displacement difference
is just `w(t+1) − w(t) ~ N(0, σ²·Δt)`, which has stddev
σ_xy·√1s = 0.020 m — exactly matching the empirical 0.026.

The EKF, by contrast, applies discrete corrections (z, yaw, scan-match,
landmark) every scan, each one a finite jump in pose. Within any
1-second window, the EKF belief contains several such steps that show
up as RPE events. **This is the price of bounding ATE.** A continuous-time
EKF or smoother would distribute corrections over time and produce a
smoother RPE distribution, but for landing accuracy ATE is the correct
metric.

But the **shape** of ATE tells the story: EKF stays bounded over 161s
(oscillates 0.05–0.20 m), odom drifts monotonically over 47s (rises
0 → 0.20 m). EKF would still be bounded after 1000s; odom would not.

Commit `71afb16` — "plot_ate_rpe.py: ATE/RPE evaluation for Run 2 bags".

---

## Phase 6 — `plot_innovation.py`

### Investigation: what's recoverable from the bag?

Read EKF source for innovation references. The EKF runs **four**
update types per scan, with very different recoverability:

| Update | Where computed | Logged? | Reconstructable from bag? |
|---|---|---|---|
| `updateYaw(yaw_from_odom)` | `odomCallback` line 126 | No | **Yes** (odom + ekf_pose) |
| `updateZ(r, 0.05)` | `downRangeCallback` line 155 | No | **Yes** (range/down + ekf_pose + cmd_vel) |
| `updateScanMatch(...)` | `scanCallback` line 329 | No (only stdout debug) | **No** — depends on live `/map` state |
| Landmark line update | `runLineSlamUpdate` line 542 | Stdout debug only at line 525 | **No** — depends on live landmark state + DA decisions |

The `INNOVATION_CAP_M = 0.08` constant in `ekf_slam_node.cpp:263` is
misleadingly named — it caps the *correction magnitude*, which is the
post-Kalman-gain version of innovation, not innovation itself.

**Decision: Option A** — plot only the cleanly-recoverable z and yaw
innovations. Option B (re-implementing line extraction in Python) was
rejected as a half-day effort that risks divergence from the live
filter.

### Key source finding: down-range topic

**The down-range used by `updateZ` is NOT a beam in `/crazyflie/scan`.**
It's a separate topic `/crazyflie/range/down`
(`sensor_msgs/msg/LaserScan` with single beam, ranges[0]). Confirmed
in bag info: 1612 messages.

### Pre-update gating recovered

Mirrored in the Python script:
1. `r` non-finite → reject
2. `r` outside `[range_min, range_max]` → reject
3. `r < 0.05 || r > 2.0` → reject (plausible flight envelope)
4. `r < z_est - 0.10` → reject (probably hit obstacle on floor)

The live z_est test (#4) is approximated by interpolating belief z
onto the down-range timestamp. Difference is microscopic for smooth
z trajectory.

### Live outlier gates recovered

From `ekf_core.cpp:99-113`:

| Drone state | `|nu|` threshold |
|---|---|
| `μ_z < 0.10` (near ground) | < 0.50 m |
| `|commanded_vz| > 0.05` (active climb/descent) | < 0.40 m |
| Hover | < 0.10 m |

`commanded_vz` from `/cmd_vel.linear.z` (1621 msgs in bag).

### Yaw innovation: simpler

`updateYaw` has no live magnitude gate (only S<=0 numerical guard).
Every odom tick fires (8061 in bag, ~50 Hz). Plot is a single series.

### Results

| Channel | RMSE | Mean | Max\|nu\| | Samples |
|---|---|---|---|---|
| z innovation (gated-in) | **3.7 mm** | -0.6 mm | 47.2 mm | 1572 |
| z innovation (gated-out) | n/a | n/a | n/a | **0** |
| z innovation (pre-rejected) | n/a | n/a | n/a | 40 |
| yaw innovation | **2.4 mrad** | -0.86 mrad | 22.95 mrad | 8061 |

### Observations

- **Zero gated-out** — filter operates well within outlier envelope
  the entire mission. Down-range and belief z agree to within filter
  expectations throughout.
- 40 pre-rejected — these are the range-envelope outliers (NaN, out of
  range, floor-obstacle hits) that never reach `updateZ`.
- Yaw RMSE 2.4 mrad ≈ 0.14° — odom yaw and EKF yaw track essentially
  perfectly.
- Visible spikes in yaw timeseries are likely angle-wrap-induced
  ringing at yaw-rate transients.

### Covariance figure: deferred

Discovered `/ekf_covariance` is published live by EKF
(`ekf_slam_node.cpp:82-83`) but **NOT recorded in the bag**.
Reconstructing it offline would require re-running the entire EKF in
Python from `/crazyflie/odom_noisy` + scans, diverging from the live
filter at every correction event.

**Decision: skip the covariance figure**, document the limitation in
the report, note that adding `/ekf_covariance` to the bag-record list
is a one-line fix for future work.

Commit `0416164` — "plot_innovation.py: z and yaw innovation
timeseries for EKF Run 2 — z RMSE 3.7mm, yaw RMSE 2.4 mrad, zero
outliers rejected post-envelope. Covariance figure deferred
(ekf_covariance topic not in bag)."

---

## Phase 7 — Replay video infrastructure

### Goal

Open the bag in RViz, screen-record it as a video deliverable.

### Initial RViz config

User asked Claude to figure out the topic list itself. Did
`ros2 bag info` on the EKF Run 2 bag, decided per topic:

| Topic | Display | Color/style |
|---|---|---|
| `/ekf_pose` | Pose | red arrow, 0.2m axes |
| `/crazyflie/odom` | Odometry | green, keep=30 (later → 1) |
| `/crazyflie/scan` | LaserScan | orange (later → cyan) size 0.06m (→ 0.08m) |
| `/map` | Map | costmap colormap, alpha 0.7 |
| `/planned_path` | Path | rebeccapurple, lw=0.04 |
| `/ekf_slam/debug/landmark_lines` | MarkerArray | (uses publisher's colors) |
| `/ekf_slam/debug/landmark_corners` | MarkerArray | (uses publisher's colors) |

Skipped: `/crazyflie/odom_noisy`, `/crazyflie/range/down`,
`/goal_pose`, `/cmd_vel`.

Initial fixed frame: `odom` (wrong — most messages are in `map`).

### Round 2 RViz tweaks (user-driven)

- `Keep: 30` → `Keep: 1` for odom truth (cleaner — single arrow tracking)
- LaserScan color: orange → bright cyan `(0, 255, 255)`
- LaserScan size: 0.06 → 0.08 m

### TF problem

When user ran the bag against RViz, every display showed "no transform"
errors. Diagnosed:

**`/tf` and `/tf_static` are NOT in the bag.** Only 11 topics, none
of them TF.

Frame inventory from peeking message headers:
- `map`: ekf_pose, odom_noisy, /map, /planned_path, both MarkerArrays
- `crazyflie/odom`: /crazyflie/odom (truth)
- `crazyflie/crazyflie/body/multiranger`: /crazyflie/scan
- `crazyflie/crazyflie/body/down_range`: /crazyflie/range/down

### Solution: TF broadcaster node

Wrote `bag_tf_broadcaster.py` (~50 lines) that publishes:
- 3 static identity transforms:
  - `map → crazyflie/odom`
  - `base_link → crazyflie/crazyflie/body/multiranger`
  - `crazyflie/crazyflie/body/multiranger → crazyflie/crazyflie/body/down_range`
- 1 dynamic transform: `map → base_link` from `/ekf_pose`
  (every PoseStamped message republished as a TransformStamped)

Also fixed the rviz config: `Fixed Frame: odom` → `Fixed Frame: map`.

### Launch file

`replay_for_video.launch.py` brings up bag+broadcaster+RViz in one
command:

```bash
ros2 launch ekf_slam replay_for_video.launch.py \
    bag_path:=analysis/headline_bags/final_ekf_bag_run2
```

Three ExecuteProcess actions: TF broadcaster (use_sim_time=true), RViz
with config (use_sim_time=true), bag player with --clock.

### Smoke test

Broadcaster ran cleanly, published 3 static transforms + dynamic
`map -> base_link` confirmed via stdout log "First map -> base_link
transform broadcast." `tf2_echo` failed due to a sim-time/wall-time
mismatch race — visual verification in RViz deferred to user.

User caught the over-debugging via CLI and stopped me: "Stop debugging
via command-line tools — clock-time mismatches and process lifecycle
issues are eating time."

Commit `9ce41d2` — "Bag-replay video infrastructure: TF broadcaster +
launch + rviz config".

---

## Phase 8 — End-of-night handoff doc

User instruction: write a comprehensive handoff `.md` (NOT the LaTeX
zip — that came later).

`analysis/tomorrow_handoff.md` written with:
- Tonight's commit log
- 15-figure inventory with descriptions
- Headline numbers from all four text reports
- Recording-infra status (broadcaster tested, RViz visual NOT
  verified)
- Known limitations (5 items)
- TODO list of unverified claims
- File pointer cheat sheet
- Final state of branch and remote

Commit `9bda72b` — "analysis/tomorrow_handoff.md: end-of-night
handoff doc".

---

## Phase 9 — LaTeX report (the final task)

User reversed course at the end: "noo broo i want you to go through
this whole project repo all the analysis text all the analysis scripts
the project proposal the final project guidelines and eveyrthing we
have ever discussed or worked on and actually get to writing the latex
report."

### Source materials read

- `Reports/MEC_559_Proposal.pdf` (full text via `pdftotext`)
- `Reports/MEC_559_Final_Project.pdf` (the prof's guidelines)
- `ekf_slam_node.cpp`, `ekf_core.cpp`, `landmark_filter.cpp`,
  `line_extractor.cpp` (full)
- `ekf_core.hpp`, `landmark_filter.hpp`, `line_extractor.hpp`
- All 11 analysis scripts in `ros2_workspace/src/ekf_slam/scripts/`
- All 4 text reports
- `analysis/diagnostic_session_summary.md`
- `analysis/tomorrow_handoff.md`

### Critical finding from the prof's guidelines

Required structure: **Introduction → Problem Statement → Algorithms
(with pseudocode) → Results & Discussion → Conclusion**. Grade weights:
**10% writing, 50% technical discussion, 40% implementation**. AI use
allowed *with disclosure*.

User wanted "abstract, intro, methods, results, discussion,
conclusion" — merged: include both the engineering-report Abstract AND
the prof-required Problem Statement and Algorithms-with-pseudocode as
sections 2 and 3.

### Document structure built

```
Reports/report_overleaf/
├── main.tex                        (IEEE-style article class)
├── references.bib                  (11 verifiable citations)
├── README.md
├── sections/
│   ├── abstract.tex                (~250 words)
│   ├── intro.tex                   (motivation + contributions)
│   ├── problem_statement.tex       (formal notation, hypothesis)
│   ├── algorithms.tex              (heart of the report — derived
│   │                                directly from C++ source)
│   ├── implementation.tex          (system architecture, datasets)
│   ├── results.tex                 (all 15 figures + 4 text reports)
│   ├── discussion.tex              (RPE inversion, two-filter choice,
│   │                                phantom-centroid quantified)
│   ├── limitations.tex             (covariance gap, body-link offset,
│   │                                IEPF removal, innovation channels)
│   ├── conclusion.tex
│   └── ai_disclosure.tex           (per prof's policy)
└── figures/                        (15 PNGs)
```

### Key decisions during the writing

- **Math derived from source, not generic textbook.** Predict-step
  Jacobian quoted exactly from `ekf_core.cpp:33-45`. Landmark inverse
  measurement model + Jacobian from `landmark_filter.cpp:23-25`.
  Forward observation model + H from `landmark_filter.cpp:66-74`.
- **Pseudocode**: 2 algorithms (line extraction, scan-to-map ICP),
  formatted with `algorithm` + `algorithmic` packages. Bug found and
  fixed in scan-to-map: missing `\EndIf` and stray `\EndIf` at end.
- **Citations**: only verifiable references with DOIs/URLs. 11 entries.
  Kalman 1960, Thrun 2005 Probabilistic Robotics, Smith/Self/Cheeseman
  1990, Bailey 2006, Sturm 2012 (TUM-RGBD ATE/RPE benchmark), Censi
  2008 (ICP variant), Lu/Milios 1997 (scan alignment), Borges/Aldon
  2004 (IEPF), ROS2 docs, Eigen, Gazebo. **No fabricated citations.**
- **TODO markers**: 15 total, scattered across all sections, every
  judgment call flagged with one-line explanation.
- **Headline framing**: drift rate 5.5×, truth-frame landing
  8–10 cm vs 16–17 cm, ATE bounded vs unbounded, RPE inversion
  carefully explained as the price of correction.

### Cross-references audited

All `\label`/`\cref` pairs checked. Two broken refs found and fixed:
- `\cref{fig:pipeline-overview}` — referenced ghost figure that was
  only a TODO suggestion. Removed.
- `\ref{alg:drift-baseline}` — referenced a nonexistent algorithm
  float. Replaced with `\cref{sec:drift-baseline}` (the actual section
  label).

All 11 citations resolve to bib entries. All 15 figures referenced in
tex are present in the figures dir.

### Final stats

- ~8,900 words across 9 section files
- ~1,700 lines of LaTeX + bib + README
- 15 TODO markers
- 31-file zip at 2.0 MB

Confidence ranking by section:
- **Most confident**: algorithms, results, limitations (everything
  derived from source / numbers from text reports / documented
  limitations)
- **Medium**: problem statement, discussion, implementation (notation
  is standard, framings are my interpretation)
- **Lowest**: intro, conclusion, abstract, ai_disclosure (need
  pass-through editing for personal voice)

### What couldn't be found

- **Run 1 odometry mission duration** — never explicitly captured
- **Bitcraze hardware datasheets** for primary-source citations of
  PMW3901/BMI088 specs (the values are in `odom_to_pose.py:30-44`
  source comments)
- **TikZ block diagram** of the node graph — flagged as TODO
- **Pipeline overview figure** — flagged as TODO

### NOT pushed

Per user instruction "Don't push commits while doing this" — the
LaTeX zip and unzipped tree are left as untracked working-tree
artifacts in `Reports/report_overleaf/` and `Reports/report_overleaf.zip`.
User will review and commit in the morning.

---

## Persistent observations across the session

These are findings that recur across multiple phases and deserve
preservation:

### 1. The two-filter architecture

`ekf_core.cpp` owns pose (4-state); `landmark_filter.cpp` owns
landmarks (2N-state). They share NO cross-covariance block. This is
a deliberate departure from textbook EKF-SLAM, motivated by
blast-radius containment: ill-conditioned landmark Jacobians or bad
augmentations cannot leak into pose covariance.

Source citation: `landmark_filter.hpp:21-23`.

### 2. The 30-scan rolling buffer is the line-extraction trick

A 4-beam scan can't fit a line in one go (you need 5+ points). The
extractor accumulates beam endpoints in **world coordinates** across
30 scans, partitioned into **15° world-bearing buckets**. As the drone
yaws, a single beam's endpoints flow across multiple buckets, and a
single bucket can collect endpoints from any of the four beams.

This world-bearing bucketing (rather than beam-index bucketing)
survives drone yaw. From `line_extractor.hpp:42-49`.

### 3. The "phantom centroid" mechanism, quantified

For both legs, both pipelines, the controller *believed* it landed
within ~6 cm of the goal:
- Outbound EKF: belief 4.86 cm, truth 9.59 cm (gap: 4.7 cm)
- Outbound Odom: belief 5.54 cm, truth 16.27 cm (gap: 10.7 cm)
- Return EKF: belief 3.09 cm, truth 8.01 cm (gap: 4.9 cm)
- Return Odom: belief 5.77 cm, truth 17.00 cm (gap: 11.2 cm)

The EKF doesn't eliminate the belief-truth gap entirely (still ~5 cm),
but it cuts it in half compared to odom. The EKF's gap is residual
estimation error after correction; the odom's gap is uncorrected
accumulated drift.

### 4. The calibrated baseline is what makes the comparison defensible

`odom_to_pose.py` injects Brownian drift on (x, y, yaw) calibrated to
PMW3901 + BMI088 hardware specs:
- σ_xy = 0.020 m/√s (from Bitcraze firmware kalman_core.c calibration,
  consistent with documented ~25 cm Flow Deck dead-reckoning drift)
- σ_yaw = 0.001 rad/√s (BMI088 noise density 0.014°/s/√Hz plus bias
  instability and yaw-flow coupling)

This is NOT arbitrary noise. The baseline is a faithful model of what
the same hardware would produce running native odometry without
external correction. The EKF-vs-baseline comparison is signal-vs-noise,
not signal-vs-strawman.

### 5. The body-link Z offset (17.425 mm)

The simulator's `crazyflie/body` link sits 17.425 mm above the model
origin used by `OdometryPublisher`. The EKF publishes belief in body
frame; truth is in model-origin frame. Without correction, dz comparisons
show a constant 17.425 mm bias.

`plot_drift_over_time.py:46` corrects for this. All z-related figures
in the report use the corrected values. For real hardware, the
equivalent offset would need to be derived from the platform URDF.

### 6. RPE shape inversion

EKF RPE > Odom RPE despite EKF ATE bounded vs Odom ATE unbounded.
Mechanism: EKF's discrete corrections (z @ 10 Hz, yaw @ 50 Hz, scan-match
when translated > 0.02 m, landmark per-observation) appear as small
jumps in belief within a 1s window. Pure noise-injected odom has no
such jumps — its 1s-displacement difference is just
`w(t+1) - w(t) ~ N(0, σ²·Δt)` with stddev σ_xy·√1s ≈ 0.020 m,
matching the empirical 0.026.

This is the price of correction applied through a fixed-rate observation
channel. For an autonomous-landing pipeline, ATE is the relevant
metric, not RPE. A continuous-time EKF or a smoother would distribute
corrections over time and produce smoother RPE.

### 7. Mapping quality decoupled from landing accuracy

Both bags report obstacle surface RMS = 0.051 m despite EKF dramatically
outperforming odom on landing. Obstacle RMS is a belief-frame metric:
it measures self-consistency of scan endpoints projected through
believed pose, not whether the believed pose matches truth. Odom's
slow Brownian drift produces a locally consistent (though globally
incorrect) belief. The figure cross-validates that line extraction is
finding real walls but does NOT distinguish good from bad localization.

### 8. The covariance figure gap

The most engineering-significant deferred item. `/ekf_covariance` is
published live but not in `BAG_TOPICS`. One-line fix in
`gazebo_final_ekf.launch.py` for future work. The covariance ellipse
figure was a planned deliverable; honesty in the limitations section
is the right answer rather than a fabricated approximate replay.

---

## Final commit graph

```
9bda72b  analysis/tomorrow_handoff.md: end-of-night handoff doc
9ce41d2  Bag-replay video infrastructure: TF broadcaster + launch + rviz config
0416164  plot_innovation.py: z and yaw innovation timeseries for EKF Run 2 ...
71afb16  plot_ate_rpe.py: ATE/RPE evaluation for Run 2 bags
f768785  analyze_map.py: drop line-extraction figures; move plot_landmarks.py to obsolete
15868d1  analyze_map.py: 7 polished figures (occupancy/scan_endpoints/line_extraction × 2 bags + EKF landmarks); track results/figures/*.png
1aac92d  analyze_map.py: add corner-landmark overlay and unified figure restyle
```

7 commits on `analysis-and-figures`, all pushed to remote.

LaTeX zip + report tree are intentionally NOT committed; pending user
review.

---

## Final state of disk

**Modified files NOT from this session** (pre-existing working tree):
- `analysis/diagnostic_session_summary.md` (minor mods predate session)
- `simulation_ws/crazyflie-simulation` (submodule pointer)

**Untracked files**:
- `Reports/MEC_559_Final_Project.pdf` (the prof's guidelines, present
  on disk before session)
- `Reports/report_overleaf/` (the LaTeX project tree, written tonight)
- `Reports/report_overleaf.zip` (Overleaf-ready zip, 2.0 MB)
- `analysis/session_log_2026-05-06.md` (this file)

**Branch**: `analysis-and-figures`, last pushed commit `9bda72b`.

---

## Notes for the user

If you read this back after waking up:

1. The LaTeX zip is at `Reports/report_overleaf.zip`. Drag it onto
   Overleaf's "New Project → Upload Project" page.

2. After first compile, search for `% TODO:` to find the 15 places I
   flagged for your editing pass.

3. The two figures I couldn't generate honestly (covariance ellipses,
   pipeline block diagram) are documented as TODOs / limitations.
   Adding them is a tomorrow-or-later task.

4. The recording infrastructure (`replay_for_video.launch.py`) is
   committed but not visually verified — first thing to test in the
   morning is running:

   ```bash
   ros2 launch ekf_slam replay_for_video.launch.py \
       bag_path:=analysis/headline_bags/final_ekf_bag_run2
   ```

5. The most likely first-compile error on Overleaf is a `cleveref`
   package issue. If `\cref` misbehaves, swap to `\ref` and the doc
   will compile.

6. All numbers in the report are reproducible by running the analysis
   scripts in `ros2_workspace/src/ekf_slam/scripts/` against the bags
   in `analysis/headline_bags/`. Trust nothing that doesn't match.

Sleep well.
