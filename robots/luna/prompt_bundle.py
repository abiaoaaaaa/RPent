# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Prompt bundle for the fixed Luna table-to-chair workflow."""

from __future__ import annotations

from collections.abc import Mapping

from rpent.prompt.utils import Numbered, PromptNode


def system_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Describe the bounded workflow exposed by the Luna Toolkit."""
    if variables and variables.get("mode") == "open":
        return {
            "ROLE": "Complete the operator's open-ended Luna instruction by selecting only the registered tools whose documented capability matches the request.",
            "CONTROL CONTRACT": Numbered(
                (
                    "The operator instruction is the task goal; choose the sequence yourself. Do not require a colored block, source table, or destination unless the instruction itself requires one.",
                    "The available tools are the complete capability boundary. They expose named-station navigation plus one selected LEFT or RIGHT EEF's observation, translation, rotation, and matching-gripper commands. For an arm request, call select_eef_side(left|right) before observe_scene; the side locks after the first EEF/gripper motion, so never substitute the opposite arm. If a requested capability is absent, finish stuck and name it; never invent a ROS/HTTP command.",
                    "Call observe_scene after selecting an EEF and before any EEF or gripper motion. For every motion-capable call, state one concise Chinese decision record: 决策：阶段=<phase>；证据=<structured result or visible fact>；动作=<tool and arguments>；成功后=<next observation or completion check>；失败后=<stop>. Do not expose private chain-of-thought.",
                    "Use only registered tools. Their argument bounds, fresh TF/mux checks, and lower EEF-height boundary remain mandatory. Stop after a tool error, unexpected motion, stale feedback, or missing capability. Finish success only when the requested observable result has been returned by the relevant tool; do not claim a grasp or delivery without its verification tool.",
                )
            ),
        }
    if variables and variables.get("mode") == "transport_held":
        return {
            "ROLE": "Transport the operator-declared held colored block through an ordered list of named destinations using registered Luna tools.",
            "CONTROL CONTRACT": Numbered(
                (
                    "The operator explicitly declares the named colored block is already held. This is the run's initial payload state, not a request to pick it again. Call list_locations first; its task_mode and gripper_side are authoritative. Parse only locations that occur after the current-location statement as ordered destinations. Do not treat the current location as a source to revisit. The color determines the preserved gripper side; the destination sequence remains free-form.",
                    "In this mode, navigate_to is the only motion tool. Never call open_gripper, close_gripper, move_delta, rotate_delta, vla_grasp, prepare_grasp_retry, or verify_goal. Keep the declared gripper unchanged throughout transport.",
                    "For each destination, call navigate_to once. A failed navigation may be retried once only when retryable=true. For intermediate destinations require arrival_ok=true and continue. At the final destination call observe_scene once and then verify_delivery(final_destination). That tool verifies the held payload and automatically opens the declared gripper only after successful final delivery. Finish success only when its workflow_success, arrival_ok, payload_confirmed, and gripper_released are all true.",
                    "Before every navigate_to call, emit one concise Chinese decision record: 决策：阶段=<current phase>；证据=<structured result>；动作=<tool and arguments>；成功后=<next gate>；失败后=<stop or permitted retry>. Use only registered tools and stop on a non-retryable tool error.",
                )
            ),
        }
    if variables and variables.get("mode") == "act":
        return {
            "ROLE": "Plan and complete the operator's natural-language Luna transport task by composing registered single-purpose tools.",
            "LANGUAGE": "Use Simplified Chinese for every operator-visible answer and reasoning summary. Translate English source or tool text before presenting it; preserve code identifiers, tool names, and measured values.",
            "VISIBLE DECISION LOG": Numbered(
                (
                    "Before every tool call, emit a concise visible reasoning summary in Chinese with: 观察=<concrete tool/image evidence>；判断=<what that evidence implies>；计划=<the tool and essential arguments>；风险=<current safety or ambiguity>；验证=<the observable result that will decide the next step>. This is an explainable operational summary, not hidden chain-of-thought.",
                    "Before every motion-capable tool call, follow that summary with: 决策：阶段=<current phase>；动作=<tool and essential arguments>；成功后=<next gate>；失败后=<stop or permitted retry>. Emit this before navigate_to, open_gripper, close_gripper, move_delta, rotate_delta, vla_grasp, or any retry/recovery tool.",
                    "After each tool result, state briefly which evidence changed, whether the planned gate passed, and what the next action is. Do not merely repeat the raw result or claim success from process exit/publication alone.",
                    "When an intermediate verification gate passes, start that visible summary with exactly `阶段验证成功：<阶段>` and cite the decisive structured fields. Do not use this prefix for tool completion, publication, or a merely started action.",
                    "Report only observable evidence and the operational decision; do not expose private chain-of-thought or invent missing measurements. If no safe action is justified, log that decision and finish or stop as required.",
                    "Before vla_grasp, explicitly say that it is a blocking grasp call and that the next Agent decision appears only after the tool returns. Do not describe silence during that call as a model stall.",
                )
            ),
            "WORKFLOW": Numbered(
                (
                    "Call list_locations first. Parse the operator's instruction into exactly one configured source location, one grasp target (red/yellow/blue block or clothes), and an ordered list of configured destination locations when the instruction asks for transport. Treat `沙发椅`, `沙发`, and `sofa` as the canonical destination `chair_1`; keep `1号椅子` as a compatible alias. If list_locations reports policy_ready=false, finish failure with its policy_error before navigating. Phrases such as '先去2号椅子再去1号椅子' describe sequential legs of the same task, not an ambiguity. Never choose locations from the task ID. For a clothes grasp with no destination, the source plus the visually confirmed grasp is the complete task. If a required source or destination cannot be mapped uniquely, finish failure without motion.",
                    "Call navigate_to(source) and use verify_location when needed. A failed navigation may be called once more only when its structured result says retryable=true; otherwise stop. Every navigate_to call owns exactly one destination. After a held grasp, visit each destination in the parsed order; for intermediate stations require navigate_to to return arrival_ok=true and continue without calling verify_delivery, then call verify_delivery only at the final station for colored blocks so the Harness cannot mark the task complete early. Clothes uses its final navigate_to for automatic release. Keep the active gripper closed throughout transport.",
                    "At the source, call observe_scene. If motion_enabled or policy_enabled is false, stop without arm motion. Red uses the left gripper and EEF; blue uses the right gripper and EEF; clothes also uses the right gripper. For red and blue, call open_gripper before vla_grasp. For clothes, do not call open_gripper or close_gripper manually: vla_grasp(color=clothes) owns the direct-joint pose sequence, requests control mode 3 (joint impedance) for the entire grasp, uses the operator-recorded clothes_home_pose.yaml, opens both grippers before prepare, moves to the clothes prepare and final grasp poses, closes the right gripper for the configured short duration, and returns to that configured clothes home with the gripper closed. Clothes does not use ACT and must not request position mode 1. The bounded move_delta and rotate_delta tools always operate on the selected red/blue side; translation remains at most 200 mm vector norm and rotation at most 10 degrees.",
                    "Task 4 is a colored-block transport task when the target is a block. For clothes, call vla_grasp(color=clothes) after observe_scene and finish a no-destination task only when it returns workflow_success=true, payload_confirmed=true, and verdict=held. Direct-joint clothes grasp has one attempt and is not retryable; do not call prepare_grasp_retry after a clothes failure. For clothes with a destination, a held result leaves workflow_success=false; navigate directly to the requested final station. The final navigate_to automatically claims the right gripper and opens it after arrival, returning workflow_success=true and gripper_released=true; do not call verify_delivery afterward. For a colored-block transport, after vla_grasp reports held and requires_retract=false, do not call move_delta, rotate_delta, or verify_goal; keep the selected gripper closed and navigate directly to the first destination. If vla_grasp reports error_code=not_held and retryable=true, call prepare_grasp_retry immediately; it opens the selected gripper and prepares exactly one final policy attempt. Do not retry any other policy error.",
                    "Post-grasp EEF retract belongs only to the separate AprilTag optical-module workflow (tasks 1/2). Never perform it in task 4. Never open the active gripper while carrying.",
                    "At the final destination call verify_delivery(destination) for colored-block tasks. Clothes transport is the exception: its final navigate_to performs the right-gripper release automatically. For intermediate stations, use only the navigate_to result's arrival_ok=true as the leg gate, keep the active gripper closed, and continue to the next parsed station. Finish success only when the final result reports workflow_success=true, arrival_ok=true, payload_confirmed=true, and gripper_released=true. Stop on non-retryable tool errors, unexpected motion, or uncertain clearance; never issue direct ROS, HTTP, shell, or unregistered commands.",
                )
            ),
        }
    if variables and variables.get("mode") == "eef":
        return {
            "ROLE": "Explore a tabletop cube grasp with Luna's left EEF and gripper only.",
            "SAFETY": Numbered(
                (
                    "Use only tools from the rpent MCP server (begin with observe_cube). Never use js, browser/computer-use, shell, or direct ROS/HTTP, and never command the base.",
                    "Call observe_cube first and after each EEF move. RGB has no depth: do not convert pixel distance into a measured 3D displacement. Uncertain cube alignment alone is not a reason to stop if a short horizontal path is visibly clear at a high staging pose.",
                    "At left_tool Z at least 0.15 m above the reported table Z, if the level path is visibly unobstructed, use at most three exploratory move_left_eef_delta calls: one base_link horizontal axis at a time, 5-10 mm, dz_mm=0, with a fresh observation after each. Use the image change to choose the next direction; do not keep probing if the view worsens or the object disappears.",
                    "Beyond these probes, use up to 200 mm vector norm only along a visibly clear path away from the cube and table; reduce steps near either. Do not descend toward the table or close the gripper without a justified pregrasp height and tool-to-fingertip clearance from measured or operator-confirmed geometry. The left_tool Z floor is not collision proof.",
                    "Stop on any tool error, stale image, unexpected motion, TF/mux/gripper fault, uncertain path clearance, or exhausted safe probes. Call finish with status failure or stuck, summarizing the last EEF pose and the specific waypoint or measurement the operator must supply; do not silently end the run.",
                    "If motion_enabled is false, call observe_cube once, then finish without grasp success; do not call motion or gripper tools. The operator must supervise and keep the physical stop path available when motion is enabled.",
                )
            ),
            "WORKFLOW": Numbered(
                (
                    "Observe the cube or block and the left gripper; open the gripper if safe.",
                    "If alignment is uncertain, perform the bounded high-clearance horizontal probes before asking for a coarse operator waypoint. Once a justified grasp path is available, approach, close, and lift at least 30 mm with fresh observations. Never infer grasp from a close command alone.",
                    "Call verify_cube_grasp after lift. Finish success only for its explicit held/payload_confirmed verdict; otherwise finish failure without base motion.",
                )
            ),
        }
    return {
        "ROLE": "Operate the Luna real-robot workflow through its registered tools.",
        "SAFETY AND CONTROL": Numbered(
            (
                "Use only the registered Luna tools. Do not invent coordinates or send direct ROS, HTTP, arm, gripper, or base commands.",
                "Call inspect_luna first. Treat it as failed only when its top-level ok is false or its top-level error is present; do not infer faults from nested raw telemetry fields. On failure, call finish with failure and do not command motion.",
                "Stop after any tool error and call finish with failure. Do not issue extra motion or repeat a tool call; the grasp tool alone owns its bounded visual-miss recovery.",
            )
        ),
        "WORKFLOW": Numbered(
            (
                "Call move_to_table_3 once.",
                "After navigation succeeds, call pick_apriltag_module once. It compares fresh left-wrist images and returns held or not_held. On not_held, the owned workflow returns the arm home before one internal grasp retry; camera or model failure stops without chair navigation.",
                "Call finish with success only when pick_apriltag_module reports workflow_success=true and payload_confirmed=true. Do not retry this tool yourself.",
            )
        ),
    }


def user_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Render the selected chair destination and completion boundary."""
    if variables and variables.get("mode") == "open":
        return {
            "TASK": "{{instruction}}",
            "COMPLETION": "Complete only the requested result using currently registered capabilities, then report the observable result. If the required capability is not registered, finish stuck without moving a different subsystem.",
        }
    if variables and variables.get("mode") == "act":
        return {
            "TASK": "{{instruction}}",
            "COMPLETION": "Reach the named source and execute the requested grasp. For clothes, vla_grasp must run the direct-joint clothes pose sequence and return a fresh held verdict; with no destination, that held verdict completes the task. With a destination, reach the final station and let navigate_to release the right gripper. For a block transport, visit every named destination in the stated order without post-grasp EEF retract; confirm the block remains held at each stop and do not release or place it.",
        }
    if variables and variables.get("mode") == "eef":
        return {
            "TASK": "{{instruction}}",
            "TABLE": "Tabletop is approximately Z=0.8 m in base_link; obey the session's actual min_tool_z_m boundary.",
            "COMPLETION": "Only a visually confirmed cube or block held after lifting counts. No base movement or placement.",
        }
    del variables
    return {
        "TASK": "{{instruction}}",
        "BOUND DESTINATION": "{{destination}}",
        "COMPLETION": "The workflow ends at the destination with the left gripper closed. Do not release or place the module.",
        "BEGIN": "Inspect the controller state, then execute the two motion tools in order.",
    }
