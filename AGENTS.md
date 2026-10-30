# Working on RPent

## Automatic high-quality camera restart

After a robot reboot, restore the camera stream with
`scripts/restart_quadcam_high_quality.sh`. It stops only the matching
`ros2 launch gmsl_quadcam` process on the robot, starts
`/home/nvidia/start_quadcam.sh`, and verifies one publisher, `1920x1080`,
JPEG quality 90, and a live frame rate from this workstation. It uses
interactive SSH/key authentication and never stores a password. This is not
a persistent system-service change; the factory Apex startup configuration
still needs an administrator change if high quality must survive reboot
without the wrapper.

```bash
bash scripts/restart_quadcam_high_quality.sh --confirm-static
```

## Read-only Luna VLAHost monitor

RPent includes `rpent-vlahost-monitor`, a browser monitor that polls only
VLAHost `/hil/snapshot`, keeps the most recent 30 samples, and displays the
22 joint/gripper values, a selected-joint history plot, and the top/left-wrist/
right-wrist crops. It is intentionally independent of the Dashboard task
session and sends no ROS, mux, navigation, gripper, or action requests. The
default source is the deployed robot-side service at `http://6.6.7.100:8011`;
override it with `--vlahost-url` when needed. Start it from the RPent checkout
after activating `.venv`:

```bash
rpent-vlahost-monitor --vlahost-url http://6.6.7.100:8011 --host 0.0.0.0
```

Open the printed URL from the operator workstation. This is telemetry evidence
only; it does not prove policy execution or physical motion success.
The main Dashboard's three-view panel uses a higher-contrast blue/green/orange
palette. `top` occupies the left column; `wrist_L` and `wrist_R` are stacked
vertically in the right column. Luna VLAHost rotates only the 2x2 source's
top-left head tile 180 degrees before constructing the stable 1280x2400
snapshot; do not apply an additional browser-side top rotation.
When the Dashboard is launched with `--luna-vlahost-url`, `/api/live-camera-frame`
proxies `/hil/snapshot` and the three canvases refresh every 250 ms. The panel
therefore remains live even without a new Agent step; saved step images remain
evidence artifacts rather than the live monitor source. Visible reasoning
summaries and tool call/result groups are expanded by default so operators can
inspect observations, decisions, risks, and verification gates.
Codex SDK reasoning summaries are operator-facing text: Chinese summaries are
shown as-is, while English-only raw summaries are replaced by a concise Chinese
fallback. The next visible decision record carries the concrete observation,
judgment, plan, risk, and verification gate. The Luna task prompt explicitly
requires Simplified Chinese for all visible answers and reasoning summaries.
`navigate_to` verification captures one fresh VLAHost camera frame after pose
validation and records it as `luna_navigate_<station>_verification.png`. The
Dashboard tool-result card embeds this verification evidence image under the
position/yaw result; a camera failure is reported as `visual_image_error`
without converting a valid pose check into a navigation failure.
The Luna prompt requires successful intermediate gates to start with
`阶段验证成功：<阶段>`. The Dashboard renders that text and successful tool
result cards in green; ordinary model prose and tool invocation messages retain
their normal colors.

Remote execution: read [`../../REMOTE_SERVER.md`](../../REMOTE_SERVER.md) before
using `ssh zkrh-56`. Use only `/ssd/xlb` for user data and outputs. A remote
RPent checkout has not been confirmed; locate it read-only and do not assume the
local and remote trees are synchronized.

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions, including
module ownership and interface contracts, and for testing and pull requests.
Use the implementation and its callers to verify behavior; report disagreements
with documentation instead of assuming either is correct.

## Repository map

For Luna, `robots/luna/` owns the Harness verdict and
`test.incoming_20261021/` owns ROS motion.
Task 3 is a separate CLI-only EEF cube/block exploration: `robots/luna/eef.py`
owns its tool interface and the persistent `luna_eef_session.py` child reuses
`interactive_left_tool_pose.py` for fresh TF, mux checks and continuous arm
target publishing. Its tool list deliberately excludes base navigation.
Motion is disabled unless `--luna-eef-execute` is supplied. The default
`base_link` tabletop Z=0.8 m plus 0.06 m left-tool clearance is an experimental
target bound, not a collision model; see `robots/luna/README.md` before real
execution. Offline tests do not establish physical grasp success.
Task 3 gripper preflight permits other registered command publishers, as the
existing Tag grasp script does. It still requires a command subscriber, fresh
gripper feedback and normal status; the arm requires the live mux/TF gates.
Operators must ensure VR/VLAHost are not actively issuing conflicting gripper
commands before enabling motion.
Task 3 prompt treats visual alignment uncertainty as a reason for up to three
high-clearance, single-axis 5-10 mm horizontal probes, not immediate failure.
Unclear path clearance, stale/invalid feedback, or missing pregrasp geometry
still stops motion and requires an explicit `finish(failure|stuck)` summary.
Task 4 is one CLI-only natural-language colored-block transport task; source,
color, and destination come from `--luna-instruction`, not the task ID. The
Agent composes single-location navigation, policy grasp, bounded EEF retract,
and delivery verification. `robots/luna/navigation.py` owns one-station calls
to the validated ROS navigation script, arrival checks, a run-wide cable yaw
reference, and at most one Agent-requested retry per location. Backend `act`
(default) uses independent red/blue arm16 `100000` checkpoints in
`robots/luna/config/act_models.yaml`; yellow is unsupported by ACT. Backend
`pi05` uses the single language-conditioned red/yellow/blue checkpoint and
`robots/luna/config/pi05_rgb.yaml`. ACT observes 22-D state/velocity but outputs
14 arm joints plus two grippers; the six body joints are held at measured
positions. Pi0.5 uses the corrected 149999-step `joint22_tplus2` checkpoint and
retains its 22-D action contract. The separate
run-level `--planner`, `--model`, and `--base-url` settings are also the single
source for Luna grasp-image verification: Codex runs use the configured Codex
model, while API runs reuse the same OpenAI-compatible multimodal model and
endpoint as the Agent. Do not add a second Luna-only VLM selector.
Task 4 recognizes `抓衣服`/`clothes` as a direct-joint pose sequence, not an ACT
rollout. The Agent still calls `vla_grasp("clothes")`; that branch executes
`test.incoming_20261021/clothes_pose_sequence_test.py --home-mode direct`.
The script requests control mode `3` (joint impedance) for the entire direct
sequence and never requests position mode `1`. It records the current
arm/body/head pose in `robots/luna/config/clothes_home_pose.yaml` instead of
calling the QP `go_home` service, switches the joint mux to user source `2`,
opens both grippers, moves through the clothes prepare/final poses, closes only
the right gripper for a configurable short duration (default 2 s), and returns
to the configured home joints with that gripper held. RPent then performs the
existing fresh wrist-image held verdict.
A clothes instruction with a destination continues to base navigation after a
held verdict; the final `navigate_to` opens the right gripper. A no-destination
instruction remains grasp-and-confirm. The historical clothes ACT checkpoint
and dataset remain for traceability but are not loaded by task 4. Red/blue
compact ACT and Pi0.5 paths are unchanged.
The canonical station `chair_1` is the sofa chair. Operators may say `沙发椅`,
`沙发`, `sofa`, or the compatible name `1号椅子`; task 4 maps all of them to
`chair_1` and releases the right gripper only after final arrival. `chair_2`
remains the ordinary second chair.
The separate
`lerobot_vlahost/src/lerobot/robots/luna_http/` driver now maps Luna's
`/hil/snapshot` and `/action_chunk` interfaces, and RPent preflight requires
its deployment config and exact training joint order. ACT chunks use a
10-frame measured-state smoothstep at each boundary; grippers are not blended.
Each EEF translation call is bounded by a 200 mm three-dimensional vector norm;
this is an input bound, not collision checking, so prompts still require a visibly
clear path and smaller steps near the table or object. Mocked HTTP tests and
ACT/OpenPI deployment dry-runs passed, but the pi0.5 checkpoint has not been
loaded for a live rollout and real motion remains untested.
Colorless instructions enter open-tool mode instead: named-station navigation
plus one explicitly selected left/right EEF and matching-gripper primitives are
exposed. The Agent must call `select_eef_side(left|right)` before observation;
the choice locks after the first EEF/gripper motion, so never reinterpret a
right-arm request as left-arm motion or switch sides mid-action. Policy
execution flags select no deployment in this mode,
while `--luna-eef-execute` remains the physical-motion opt-in.
Run task-4 policy deployment with the Python 3.12 environment at
`~/miniconda3/envs/lerobot_xlb`; keep its matching `lerobot-rollout` entry
point and prepend this workspace's `lerobot_vlahost/src`. Do not fall back to
the `lerobot_vlahost` Python 3.14 environment: Draccus 0.8.0 fails while
building the rollout CLI there.
The Luna Dashboard accepts `/rpent-task 4 <自然语言指令>` in its composer. It
shows the Agent decision/tool transcript and, after a Luna tool result,
read-only top/left-wrist/right-wrist frames from one fresh VLAHost mosaic.
The Dashboard never enables motion by itself: task-4 policy and EEF motion
still require the matching CLI execution flags used to launch the session.
Its free-text task field is terminal; do not add positional fields after it.
The planner can list/navigate named locations, observe, stage the selected EEF
using bounded translation/rotation and gripper tools, then call a color-selected
policy `vla_grasp`. `robots/luna/act.py` owns policy preflight and a finite `deploy.py`
process while preserving the existing ACT path. Red maps to the left gripper,
left wrist, and `left_tool`; blue maps to the right gripper, right wrist, and
`right_tool`. The owned EEF session selects exactly one matching side per run,
checks that side's TF, and fills only its matching IK hand field. Both colors
must return the selected tool to the recorded high-clearance pose and pass
`verify_goal` before destination navigation. The EEF publisher must stop before
the policy starts. Red/blue compact policies switch back after exit: a new EEF
session explicitly selects IK source 0 and requests
`/tj/control/set_input=1`, verifies both reported mux sources are 0, then
resumes the selected gripper hold before wrist VLM verification. Clothes is the
full-22D exception only during its finite grasp rollout; after a held verdict
it permits base-only transport. Its retry preparation and final destination
release claim source 0 to open the right gripper. A first
`not_held` result from either the initial policy verdict or post-retract
`verify_goal` may be retried once. Call `prepare_grasp_retry` first: it proceeds
only when the selected left/right EEF is near the recorded high pose and
otherwise reports actual and target poses for a nonzero reposition. The retract
check is only for colored blocks; clothes has no retract check and releases
automatically from its final `navigate_to`. Physical motion requires
`--luna-eef-execute` plus
the matching `--luna-act-execute` or `--luna-pi05-execute` flag. A clean policy
exit is not task success. This handoff and physical grasp remain unverified on
Luna; never allow competing action publishers.

For tasks 1/2, the grasp child emits a compact
`RPENT_RESULT=` record; a zero exit status is only script completion, not
task completion. The current `solved` gate requires a fresh VLAHost
before/after camera mosaic, an explicit `held` VLM verdict, script completion,
verified chair pose, and a ready left gripper. The image verdict is perceptual
evidence, not a physical payload sensor. The grasp child remains active and
publishes arm/gripper hold commands while the Harness performs this check;
never split the process at retract without replacing that hold ownership.
The VLAHost mosaic is cropped to the left-wrist view before the VLM call;
the head, right-wrist, and bottom views are not shown to the model. The binary
verdict is `held` when the module is visibly gripped after retract, or
`not_held` otherwise; tabletop clearance is not required. `not_held` triggers
the owned grasp recovery:
call ROS `control/go_home`, wait its reported plan duration plus one second and
check live TF/gripper feedback, then recapture Tag and retry once. Camera/model
errors and other motion faults stop without retry; a second
`not_held` returns home and stops. The go_home Trigger only acknowledges a
start, so this wait is not an independent measured home-pose verification.
The local task-bound grasp script incorporates the remote
`liuqian@6.6.7.101:/home/liuqian/copy/test` Tag grasp gates and full calibrated
approach target (no 5 mm feedback-relative approach clamp). The standalone
script defaults to a 70-degree target-angle limit; the Harness explicitly
passes 90 degrees for its bound task while retaining chair transport and
Harness result reporting; do not replace it wholesale with
the remote grasp/place/return script. The local named-station navigation already
uses the remote cable-reference yaw route and half-turn boundary; keep its
`pos.yaml` source of station coordinates. Harness table and chair navigation
run in separate processes, so the runner captures the original cable-reference
yaw before table navigation and passes it through the grasp child to chair
navigation. Do not silently re-zero at the table.
Transient stale world poses are retried read-only for at most 10 seconds in
`inspect_luna` and before locking the table-navigation cable reference; invalid
frames, disconnected feedback, and other faults are not retried. The navigation
child still requires continuously fresh odometry before commanding motion.
Harness table/chair navigation explicitly enables one bounded odometry recovery
per translation/yaw phase:
the VLAHost translation is cancelled or yaw receives confirmed zero speed,
then navigation waits at most 10 seconds for fresh, continuously updated poses
(2-second stable window) before recomputing the target. A second outage in
that phase fails.
The 0.5-second active-control freshness threshold remains unchanged; do not
drive from a stale pose or silently widen that threshold.

- `rpent/planner/`, `rpent/session/`, and `rpent/prompt/`: planner adapters,
  execution sessions, and shared prompt construction.
- `rpent/tools/` and `rpent/memory/`: tool execution, observations, and memory.
- `rpent/robots/`: robot discovery, descriptors, shared runtime, and Env/VLA
  components. `robots/<robot>/` owns each concrete integration.
- `rpent/cli/`, `rpent/dashboard/`, and `rpent/evaluation/`: entry points,
  interactive execution, and run results.
- `tests/`: offline tests and integration checks; see [tests/README.md](tests/README.md).
- `docs/source-en/` and `docs/source-zh/`: paired user and developer documentation.

## Development principles

- Confirm the working branch and relevant upstream state before editing. Read
  the closest current implementation and both sides of changed interfaces.
  Preserve unrelated working-tree changes.
- Reuse existing configuration, registration, RPC, runtime, and artifact
  helpers. Prefer RLinf's supported Env/VLA implementations with thin RPent
  adapters.
- Base compatibility paths and new abstractions on actual consumers or
  supported data.
- Validate external inputs at the boundary that owns the operation. Keep
  internal control flow direct, avoid repeated checks of established state,
  and preserve useful errors. Make resource ownership and cleanup explicit.
- Keep optional simulator/model imports at their use sites so basic imports,
  robot discovery, and CLI help work without every robot extra installed.
- Follow neighboring code for naming and structure, use the project logger,
  and follow CONTRIBUTING's type annotation and docstring conventions.
- Write prose about current behavior and non-obvious constraints. Keep review
  exchanges and implementation history in PR discussions or migration notes.
  Translate meaning into natural Chinese and preserve technical identifiers.
- Keep prompts and tool descriptions focused on the model's task. Remove
  repetition without dropping required behavior or operational constraints.
- Test observable behavior and concrete regressions, reusing existing cases
  and fixtures where possible. Avoid tests that merely restate implementation
  details. Never weaken failing tests to make a change pass.

## Task skills

Read the relevant skill when the task calls for it; ordinary edits do not
require loading every skill.

| Task | Skill |
| --- | --- |
| Explain and review a PR, or review an explicitly selected local diff | [review-pr](.agents/skills/review-pr/SKILL.md) |
| Check documentation, comments, translations, or user/model-facing prose | [docs-check](.agents/skills/docs-check/SKILL.md) |
| Select and run checks for a change, or prepare validation evidence | [verify-change](.agents/skills/verify-change/SKILL.md) |
| Add a robot integration or extend its Env/VLA/tool/runtime wiring | [add-robot](.agents/skills/add-robot/SKILL.md) |

Reuse repository tools and the commands documented in CONTRIBUTING and the
test guide. Report what ran, its result, and what remains unverified; keep
integration evidence distinct from benchmark task success.

Skill sources live in `.agents/skills/`. `.claude/skills` links to that whole
directory, and `CLAUDE.md` links to this file. Edit the canonical files so both
clients share the same instructions. Keep each skill self-contained in its
`SKILL.md` and link to existing repository resources.
