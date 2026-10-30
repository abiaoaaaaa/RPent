# Luna table-to-chair Harness

This integration exposes the physically tested Luna workflow through RPent. It
does not replace the arm, gripper, AprilTag, or mobile-base controllers. The
Harness calls the existing ROS scripts and the VLAHost HTTP service at
`http://6.6.7.101:8011`.

Task `1` transports the AprilTag 5 optical module from `table_3` to `chair_1`.
Task `2` uses the same workflow and ends at `chair_2`. Both tasks stop with the
left gripper closed. They do not shift right or release the object on success.
After an explicit visual `not_held`, the arm returns home before one bounded
grasp retry; a second `not_held` returns home and ends in failure.

Task `3` is a separate, CLI-only tabletop cube/block exploration. It exposes
`observe_cube`, `move_left_eef_delta`, `open_left_gripper`,
`close_left_gripper`, and `verify_cube_grasp`; it has no base-navigation tool.
The existing `interactive_left_tool_pose.py` provides fresh TF/mux checks and
continuous EEF target publishing. One ROS child maintains the left-arm target
and publishes left-gripper commands across tool calls. Other registered gripper
publishers do not by themselves block task 3; the subscriber, fresh gripper
feedback/status and arm mux checks still apply. Operators must ensure VR and
VLAHost are not actively sending conflicting gripper commands. A VLAHost JPEG
is used for visual feedback; it is not depth sensing or precise object
localization. The planner can choose an EEF step up to 200 mm vector norm along a verified
clear corridor, then observe again; near the cube or table it uses small steps.
The measured table height is approximately `base_link` Z=0.8 m; task 3
defaults to a left-tool target floor of 0.86 m (6 cm clearance). This is an
experimental boundary, not a collision model or proof that the jaws can safely
contact the cube.
When only image alignment is uncertain, task 3 may make up to three single-axis,
level 5-10 mm probes at left-tool Z at least 0.15 m above the table, observing
again after each. It does not probe through unclear paths or descend/close
without a justified pregrasp height and tool-to-fingertip clearance. If no safe
next action is available, it calls `finish` with status `stuck`, the pose and the
measurement or waypoint needed from the operator. TF, mux, image, motion, and
gripper faults still stop automatically.

Task 3 starts observation-only. First verify the camera, left-tool pose and
control ownership with the robot supervised:

```bash
source .venv/bin/activate
rpent --robot luna --task-id 3 --planner codex \
  --memory-profile local --memory-dir ./memory/luna --max-turns 80
```

Only after checking the measured table geometry, tool-to-fingertip offset,
clear path, controller ownership, and physical stop path, add
`--luna-eef-execute` to enable left-arm and gripper commands. Task 3 never
commands the base. `--luna-table-z-m` and `--luna-tool-clearance-m` configure
the experimental lower target bound; clearance cannot be set below 0.05 m.
After closing, a measured EEF lift of at least 30 mm and a left-wrist VLM
`held` verdict are required for Harness success. The VLM verdict is only
visual evidence, not force or payload sensing. Closing the Harness run stops
the owned EEF publisher; it does not promise continued posture holding after
the process exits. A supervised 15 mm upward EEF test measured 14.30 mm of
movement; full task-3 grasp has not been verified.

## Language-directed policy tasks (task 4; not yet physically validated)

Task `4` accepts one complete operator instruction through `--luna-instruction`,
such as `去3号桌子拿红色方块去1号椅子`. Red, yellow, blue, and clothes are
**not** separate task IDs, and source/destination are not selected by the task ID.
An instruction containing `抓衣服` or `clothes` selects the dedicated
direct-joint clothes pose sequence. With no destination it is
grasp-and-confirm; with a destination it continues base-only transport and
releases the right gripper automatically at the final named station. An
instruction without exactly one supported target
instead enters **open-tool
mode**. The Agent may compose only implemented tools: named-station navigation,
fresh three-view/selected-EEF observation, bounded selected-EEF
translation/rotation, and selected-gripper open/close. The ROS session selects
exactly one side before the first EEF/gripper motion: the Agent calls
`select_eef_side(left|right)`, then that choice locks for the run. It
reads the corresponding wrist TF, fills only the matching `IKRequest` hand
field, and publishes only the matching availability bit. It must never use the
left arm as a proxy for a right-EEF Cartesian request.
Policy execution switches are ignored in open-tool mode because no target selects
a policy deployment.
`--luna-policy-backend act` (the default) keeps the existing independent red
and blue arm16 ACT models; clothes no longer loads a policy checkpoint. ACT
intentionally rejects yellow. `--luna-policy-backend
pi05` selects one language-conditioned Pi0.5 checkpoint shared by all three
colors. The
planner calls `list_locations`, maps the natural-language source and ordered
destination sequence to names from `pos.yaml`, and composes one
`navigate_to(location)` call per leg. A phrase such as “先去2号椅子再去1号椅子”
is one transport task with two ordered destination legs, not an ambiguous
multiple-destination error.
`chair_1` represents the sofa chair: `沙发椅`, `沙发`, and `sofa` resolve to
it, while `1号椅子` remains a compatibility alias. `chair_2` is the ordinary
second chair.
Each call sends exactly one station to `native_nav_test.py`, which converts the
world-frame target to a body-frame relative command and calls the Apex ROS
`/move/MoveRelativeCmd` action. It verifies the resulting world pose through
VLAHost and preserves the original cable yaw reference across all legs and
retries. The planner then reads the scene,
optionally stages the left EEF with
`move_delta` (3D vector norm ≤200 mm), `rotate_delta` (≤10°), and the
target-selected gripper, then calls
`vla_grasp(color)`. For clothes, `vla_grasp("clothes")` instead runs
`clothes_pose_sequence_test.py --home-mode direct`: it uses the
operator-recorded `robots/luna/config/clothes_home_pose.yaml`, requests control
mode `3` (joint impedance), switches to user joint source `2`, opens both
grippers, moves to the prepare and final grasp poses, closes the right gripper
for 2 seconds by default, and returns to that fixed home pose with the gripper
held. It never requests position mode `1`.
The selected target must match the instruction. RGB images do not provide measured 3D object positions;
the planner may only stage along paths with defensible clearance.
Before every motion-capable tool call, the planner prints a concise Chinese
decision record containing the current phase, observable evidence, selected
tool, next success gate, and failure handling. These records are written to
both `codex_luna_t4.txt` and its raw `.stream.jsonl`; they are operational
justifications, not hidden chain-of-thought. The record before `vla_grasp`
also states that the policy call blocks until it returns.

The EEF ROS session stops publishing before the owned VLAHost `deploy.py`
process starts. Red/blue compact ACT and Pi0.5 switch back after a clean policy
exit: a new EEF session explicitly selects IK source `0` and requests
`/tj/control/set_input=1`, then requires both reported mux sources to become
`0` before resuming the selected closed-gripper hold. The direct-joint clothes
sequence needs no ACT policy handoff and permits base-only transport after a
held verdict. It has one attempt and no `prepare_grasp_retry`. At the final
requested station, `navigate_to` claims
source `0` and opens the right gripper automatically; no post-release visual
gate is required. A failed or timed-out required handoff stops before release.
Red uses the left gripper/wrist/EEF; blue and clothes use the right gripper/
wrist/EEF. Both block colors go directly to navigation after a held verdict;
task 4 never performs an optical-module EEF retract. `verify_goal` is an
intermediate gate that reports
`grasp_retracted=true` only when a fresh selected-wrist verdict remains
`held`, the tool is at least 0.15 m above the configured table, and its final
position is within 0.06 m of the initial high-clearance pose (or a high staging
pose if the initial pose was lower). The Agent may make one explicit grasp retry
after a first structured `not_held` result from either the initial policy verdict
or the post-retract `verify_goal` verdict. It calls `prepare_grasp_retry` first;
for red, that tool proceeds when the live left EEF is near the recorded high pose
and requests a nonzero reposition only when needed; for blue, it opens the right
gripper and prepares the retry without an EEF return-pose requirement.
After preparation the policy may run once more. Navigation also
allows at most one Agent-requested retry when the result says `retryable=true`.
At every intermediate destination the planner requires the `navigate_to` result
to report `arrival_ok=true` and keeps the active gripper closed. It calls
`verify_delivery` only at the final destination, which checks the destination
pose, selected gripper, and a fresh selected-wrist held verdict, then opens the
selected gripper. `workflow_success` additionally requires that release command
to succeed. A clean policy exit or gripper closure alone is not task success.
The `hold_closed` handoff commands closure; it does not independently prove
physical payload retention.

The manifest `robots/luna/config/act_models.yaml` points at the local red and
blue arm16 `100000` ACT checkpoints. Its historical clothes checkpoint entry
is retained for traceability but is not loaded by task 4. The red/blue
policies consume 22-D state/velocity and emit 16-D actions: 14 arm joints plus
two grippers. The Luna driver holds the six body joints at their measured
values. Clothes instead uses the measured joint pose files and the direct-joint
sequence described above.
The Pi0.5 config
`robots/luna/config/pi05_rgb.yaml` points at the corrected shared 149999-step
checkpoint
`/home/xlb/data/models/pi05_luna_stack_blocks_rgb_v4_joint22_tplus2/149999`
and its matching metadata-only safety dataset. Pi0.5 retains its 22-D action
contract: 14 arm joints, six body/waist joints and two grippers.
The dedicated `lerobot_vlahost/src/lerobot/robots/luna_http/` adapter reads
Luna's `/hil/snapshot` and sends its 7+7+6 joint and gripper `/action_chunk`
payload. It was tested with mocked HTTP responses; both ACT configs and the
Pi0.5 config pass `deploy.py --dry-run`. **No Pi0.5 model-load rollout or
physical grasp has been tested.** An ACT checkpoint `config.json` must identify
ACT; its action dimension may be the current compact 16-D contract or the
legacy 22-D contract, while state/velocity remain 22-D. Required camera
keys/shapes and dataset action names must match the deployment contract. The
selected YAML must use the same checkpoint,
`luna_http`, the configured VLAHost URL, `inference.strategy: base`,
`inference.return_to_initial_position: false`, and no recording. Preserve the
checkpoint's native preprocessing, action convention and frequency. Do not
copy the old cable-tie ACT configuration merely because it is syntactically
valid. RPent rejects missing or mismatched configuration before starting the
rollout. ACT loads only the instructed color model. Pi0.5 sends exactly one of
the training prompts `Pick up the red/yellow/blue block.` to the shared model.

The Harness starts `deploy.py` with
`~/miniconda3/envs/lerobot_xlb/bin/python`, pins the matching
`lerobot-rollout`, and prepends this checkout's `lerobot_vlahost/src` so the
Luna HTTP adapter is available. The `lerobot_vlahost` project environment uses
Python 3.14; its Draccus 0.8.0 rollout CLI currently fails while parsing union
typed arguments and must not be used for task 4.

Observation-only ACT check (replace the instruction with blue as needed):

```bash
rpent --robot luna --task-id 4 --planner codex \
  --luna-instruction '去3号桌子拿红色方块去1号椅子' \
  --memory-profile local --memory-dir ./memory/luna --max-turns 20
```

Before enabling a first physical rollout with the new driver, the operator must
confirm no competing ROS/VLAHost/VR action producer, keep an emergency stop
available, and explicitly opt in with both `--luna-eef-execute` and
`--luna-act-execute`. `--luna-act-duration-s`
defaults to 15 seconds and is capped at 120 seconds; it bounds runtime rather
than detecting policy completion. The model and final-state holding behavior
have **not** been tested on Luna, so a model checkpoint and deploy config alone
do not establish a safe or successful physical grasp.

Pi0.5 observation-only check (yellow shown to exercise the shared checkpoint):

```bash
rpent --robot luna --task-id 4 --planner codex \
  --luna-policy-backend pi05 \
  --luna-instruction '去3号桌子拿黄色方块去2号椅子' \
  --memory-profile local --memory-dir ./memory/luna --max-turns 20
```

Physical Pi0.5 execution additionally requires both `--luna-eef-execute` and
`--luna-pi05-execute`; do not also pass `--luna-act-execute`. The current Pi0.5
integration permits only ordinary chunk inference, not RTC. Before the first
model-load or robot test, recheck GPU processes and verify the live 22-D pose is
compatible with the merged training distribution. The rollout's joint step
guard, dataset bounds, fresh observations, active VLAHost receipt checks, and
operator stop path remain required.

Pi0.5 performs one inference-only warmup after connecting and discards its
actions before starting the configured duration. This prevents the several-
second first JAX inference from dispatching a chunk that the next fast inference
would replace after only a few physical steps. Every run writes a unique folder
under `/home/xlb/data/logs/luna_pi05_rollouts/` containing `run_manifest.json`,
`rollout.log`, `action_state.jsonl`, and, after normal completion,
`run_summary.json`. The JSONL records all 22 measured values, raw model actions,
post-limit actions, complete VLAHost receipts, clamp counts, and within/between-
chunk discontinuity metrics. It is diagnostic evidence; it does not enable
recording or change the task-success gate.

## Prerequisites

On `6.6.7.101`, start the ROS controller, cameras, AprilTag dependencies, and
VLAHost exactly as for the successful manual test. The Harness automatically
sources `robots/luna/config/remote_ros_env.sh`. That setup uses ROS domain 0,
CycloneDDS on `enp12s0`, and the minimal overlay at
`/home/xlb/data/code/code_new/Luna/rpent_ros_overlay_clean/install`.

The default workflow directory is `test.incoming_20261021`. It must contain the
navigation/grasp scripts, `pos.yaml`, the return-pose snapshot, and
the tag offset YAML. The named `chair_1`, `chair_2`, and `table_3` targets are
loaded from `pos.yaml`. Use `--luna-script-dir` if those files are installed
elsewhere.

List navigation records the starting yaw as a cable-neutral reference. The
Harness captures it before moving to table 3 and passes the same reference to
the separate chair-navigation process after grasping. Each station uses the
equivalent target yaw nearest that reference; navigation stops if continuous
yaw moves more than half a turn away from it. Standalone moves without a cable
reference stop after about 200 degrees of measured travel. Start the task only
when the external cable is physically untwisted. A standalone grasp script
with `--destination-pose` must also receive `--cable-reference-yaw-rad` from
its preceding table-navigation step.

The Tag grasp follows the October 22 script in
`liuqian@6.6.7.101:/home/liuqian/copy/test`: it publishes the full calibrated
left-tool target, first above the Tag and then at the pickup point. It no
longer limits each approach command to 5 mm from the current feedback pose.
The standalone script retains its 70-degree target-angle default; the Luna
Harness passes 90 degrees for its bound task. The stable-and-in-tolerance
closing gate and 40 mm closing deviation guard remain. The local Harness
retains chair transport, the visual hold gate, structured outcomes, and Tag-ID matching;
the remote script's release/return workflow is not copied. The remote base
navigation's cable-reference yaw routing and half-turn boundary were already
present locally; the Harness now carries that reference across its native
relative-motion calls. The old `/base_move_world` navigation path is no longer
used by task 4. No physical RPent task-4 run with this migration has been
verified.

Before enabling motion, check the remote service from the Harness machine:

```bash
curl --noproxy '*' -fsS http://6.6.7.101:8011/health
curl --noproxy '*' -fsS http://6.6.7.101:8011/base_world_pose
```

Run task 1 in a plain terminal:

```bash
source .venv/bin/activate
rpent --robot luna --task-id 1 --planner codex \
  --memory-profile local --memory-dir ./memory/luna --max-turns 100
```

Use `--task-id 2` for chair 2. The real-robot CLI requires a TTY. Keep the
operator stop path available throughout the run. Each child workflow writes a
log named `luna_move_table_3.log` or `luna_pick_to_chair_<n>.log` in the RPent
run output directory. `--max-turns` counts planner turns, including wait
messages while a tool runs; eight turns can interrupt the grasp before its
180-second arm or 360-second workflow timeout. Raising the turn cap does not
relax the motion checks or resolve a stalled arm.

The planner receives three Luna tools:

- `inspect_luna` reads `/state` and `/base_world_pose` without motion.
- `move_to_table_3` runs the validated world-pose navigation script.
- `pick_apriltag_module` runs AprilTag grasp and retracts without the old right
  shift. Before chair transport, the same ROS process keeps the arm at its
  return pose and the left gripper closed while the Harness checks a fresh
  VLAHost camera image with a vision-capable Codex model. It transports only
  on an explicit `held` verdict. On a clear `not_held`, it calls ROS
  `control/go_home`, waits for the controller-reported motion duration plus one
  second and checks live arm TF and gripper status before starting one fresh
  Tag-capture/grasp attempt. The service acknowledges motion start, not
  physical arrival; the timed wait and feedback checks are not an independent
  home-position measurement. Keep the operator stop path ready.

`inspect_luna` returns a compact machine verdict with top-level `ok`,
`connected`, `pose_ok`, and `pose_age_s` fields plus normalized left/right
gripper status. DM status code `1` is reported as `status: on, ok: true`; raw
VLAHost telemetry is not sent to the planner. The preflight rejects disconnected
feedback, invalid or mismatched world frames, `/odomAftUpdate` data older than
0.5 seconds, and a gripper not in status `1`. If only the world pose is stale,
`inspect_luna` polls read-only for up to 10 seconds; table navigation also
waits up to 10 seconds for a fresh pose before locking the cable-reference yaw.
Other faults stop immediately. The navigation script still requires continuous
fresh pose updates before it sends any motion command. The planner stops without
motion when top-level `ok` is false or top-level `error` is present.
During Harness table/chair navigation, each translation/yaw phase permits one
odometry interruption recovery:
the base motion is stopped, the script waits up to 10 seconds for continuously
updating fresh world poses (including a 2-second stability window), then
recomputes the remaining target. A second interruption in that phase or failed recovery stops
the workflow. A pose older than 0.5 seconds is never used for active control;
this does not make a 10-second-old coordinate safe to drive against. Standalone
navigation retains its default of no automatic odometry retry.

The grasp script emits `RPENT_GATE=await_visual_verdict` after retracting and
waits for a verdict on stdin. The Harness reads `/hil/snapshot` before the
grasp and at this gate, requires both JPEG frames to be fresh (receipt and
source age at most 0.5 s), and saves a labelled before/after comparison image
in the run directory. From each 1280×2400 VLAHost mosaic, it sends only the
left-wrist camera view to the vision model. The model returns `held` or
`not_held` plus visible evidence: jaw support after retract is the criterion;
an air gap beneath the module or clearance of its far end is not required.
`held` permits chair navigation; `not_held` returns home and retries grasping
once. A second `not_held` stops after returning home. Missing or stale images,
invalid model output, and model failures stop without chair navigation or
retry. The verdict is an image-based estimate, not an independent physical
sensor measurement. The model must support image input; use the configured
Codex default or configure a suitable model before running this workflow.

The grasp script emits one `RPENT_RESULT=` JSON record with `phase`,
`error_code`, and `message`; the Harness returns these fields alongside the
child log path and visual verdict. A child exit code of zero is only
`script_success`. After
transport, the Harness compares the fresh world pose with the selected chair
in `pos.yaml` (0.05 m position and 0.10 rad yaw tolerance) and checks the left
gripper status. `workflow_success=true` requires the `held` VLM verdict,
successful script, chair arrival, and normal left gripper status. The planner
never retries a tool call; only the grasp tool performs its one bounded
`not_held` recovery. Other failures stop immediately. The first and final
attempt logs are returned separately when a retry occurs.

Use `--luna-ros-setup` to select another setup script, or set
`LUNA_ROS_OVERLAY` before starting RPent when the minimal overlay moves. The
VLAHost client and child workflows bypass shell proxy variables for the direct
`6.6.7.101` connection.

RPent cancellation sends `SIGINT` to the owned workflow process so the existing
script cleanup stops its active base command and ROS node. A physical emergency
stop remains the operator's responsibility.
