# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dual-arm bowl stacking task for the DOS-W1 robot.

This task intentionally keeps the final success signal configurable. For real
robot RL the preferred first-stage setup is human-confirmed success via the
leader-follower keyboard wrapper. A calibrated joint-space proxy can also be
enabled for smoke tests and early automation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from rlinf.envs.realworld.dosw1.dosw1_env import (
    NUM_JOINTS,
    ControlMode,
    DOSW1Config,
    DOSW1Env,
)


def _left_grasp_joint() -> np.ndarray:
    return np.array([-0.75, 0.0, 0.0, 1.57, 0.0, -1.57], dtype=np.float64)


def _right_grasp_joint() -> np.ndarray:
    return np.array([0.75, 0.0, 0.0, -1.57, 0.0, 1.57], dtype=np.float64)


def _left_stack_joint() -> np.ndarray:
    return np.array([-0.65, -0.15, 0.0, 1.57, 0.0, -1.57], dtype=np.float64)


def _right_stack_joint() -> np.ndarray:
    return np.array([0.65, -0.15, 0.0, -1.57, 0.0, 1.57], dtype=np.float64)


@dataclass
class StackBowlsConfig(DOSW1Config):
    """Configuration for a DOS-W1 dual-arm bowl stacking task."""

    task_description: str = "Stack the bowls using both hands."

    target_left_grasp_joint: np.ndarray = field(default_factory=_left_grasp_joint)
    target_right_grasp_joint: np.ndarray = field(default_factory=_right_grasp_joint)
    target_left_stack_joint: np.ndarray = field(default_factory=_left_stack_joint)
    target_right_stack_joint: np.ndarray = field(default_factory=_right_stack_joint)

    success_mode: str = "manual"
    """One of: manual, joint, manual_or_joint.

    manual relies on the keyboard wrapper returning reward=1.0 for the success
    key. joint uses the calibrated stack joints below as a proprioceptive proxy.
    """

    use_dense_reward: bool = True
    joint_reward_sharpness: float = 2.0
    stack_joint_threshold: float = 0.18
    success_hold_steps: int = 5

    gripper_closed_max_width: float = 0.012
    gripper_released_min_width: float = 0.025
    require_release_for_joint_success: bool = True
    grasp_bonus: float = 0.2
    release_bonus: float = 0.05

    enable_gripper_penalty: bool = False
    gripper_penalty: float = 0.02

    # Defaults used by LeaderFollowerKeyboardIntervention for the stack task.
    manual_done_reward: float = 1.0
    manual_done_terminated: bool = True
    manual_done_truncated: bool = False
    manual_abort_reward: float = 0.0
    manual_abort_terminated: bool = False
    manual_abort_truncated: bool = True

    max_joint_delta: float = 0.05
    action_scale: float = 0.3
    step_frequency: float = 10.0


class StackBowlsEnv(DOSW1Env):
    """Stack bowls with both DOS-W1 arms.

    The automatic joint-space success path is a proxy and should be replaced by
    a vision/depth detector when reliable bowl pose estimates are available.
    """

    def __init__(
        self,
        override_cfg: dict,
        worker_info=None,
        hardware_info=None,
        env_idx: int = 0,
    ) -> None:
        super().__init__(
            StackBowlsConfig(**override_cfg),
            worker_info,
            hardware_info,
            env_idx,
        )
        self.phase = "reach"
        self.task_success = False
        self._success_hold_count = 0
        self._left_holding = False
        self._right_holding = False

    @property
    def task_description(self) -> str:
        return self.config.task_description

    def reset(
        self,
        *,
        seed=None,
        options=None,
        joint_reset: bool = False,
    ) -> tuple[dict, dict]:
        obs, info = super().reset(seed=seed, options=options, joint_reset=joint_reset)
        self.phase = "reach"
        self.task_success = False
        self._success_hold_count = 0
        self._left_holding = False
        self._right_holding = False
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        if self.task_success and not self.config.manual_episode_control_only:
            terminated = True
        info["success"] = bool(self.task_success or self.manual_done)
        info["phase"] = self.phase
        info["success_hold_count"] = self._success_hold_count
        return obs, reward, terminated, truncated, info

    def _calc_step_reward(self, obs: dict, gripper_changed: bool = False) -> float:
        del obs
        if self.config.is_dummy:
            return 0.0

        cfg: StackBowlsConfig = self.config
        left_joint = self.robot_state.left_joint_positions
        right_joint = self.robot_state.right_joint_positions
        left_gripper = self.robot_state.left_gripper
        right_gripper = self.robot_state.right_gripper

        left_grasp = np.asarray(cfg.target_left_grasp_joint, dtype=np.float64).reshape(
            NUM_JOINTS
        )
        right_grasp = np.asarray(cfg.target_right_grasp_joint, dtype=np.float64).reshape(
            NUM_JOINTS
        )
        left_stack = np.asarray(cfg.target_left_stack_joint, dtype=np.float64).reshape(
            NUM_JOINTS
        )
        right_stack = np.asarray(cfg.target_right_stack_joint, dtype=np.float64).reshape(
            NUM_JOINTS
        )

        left_grasp_dist = self._joint_distance(left_joint, left_grasp)
        right_grasp_dist = self._joint_distance(right_joint, right_grasp)
        left_stack_dist = self._joint_distance(left_joint, left_stack)
        right_stack_dist = self._joint_distance(right_joint, right_stack)

        left_closed = self._gripper_closed("left", left_gripper)
        right_closed = self._gripper_closed("right", right_gripper)
        both_released = (
            left_gripper >= cfg.gripper_released_min_width
            and right_gripper >= cfg.gripper_released_min_width
        )

        if left_closed:
            self._left_holding = True
        if right_closed:
            self._right_holding = True
        if self._left_holding and self._right_holding:
            self.phase = "stack"

        reward = 0.0
        if cfg.use_dense_reward:
            reach_score = self._exp_score(
                left_grasp_dist**2 + right_grasp_dist**2,
                cfg.joint_reward_sharpness,
            )
            stack_score = self._exp_score(
                left_stack_dist**2 + right_stack_dist**2,
                cfg.joint_reward_sharpness,
            )
            if self.phase == "reach":
                reward = 0.35 * reach_score
                if self._left_holding and self._right_holding:
                    reward += cfg.grasp_bonus
            else:
                reward = 0.45 + 0.45 * stack_score
                if both_released:
                    reward += cfg.release_bonus

        if cfg.enable_gripper_penalty and gripper_changed:
            reward -= cfg.gripper_penalty

        if self._joint_success(
            left_stack_dist=left_stack_dist,
            right_stack_dist=right_stack_dist,
            both_released=both_released,
        ):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0

        success_by_joint = self._success_hold_count >= max(1, cfg.success_hold_steps)
        if cfg.success_mode not in {"manual", "joint", "manual_or_joint"}:
            raise ValueError(f"Unsupported success_mode={cfg.success_mode!r}")

        if cfg.success_mode in {"joint", "manual_or_joint"} and success_by_joint:
            self.task_success = True
            return 1.0

        # Keep shaped rewards strictly below 1.0 because RLinf treats reward==1
        # as a success marker in RealWorldEnv._record_metrics().
        return float(np.clip(reward, -1.0, 0.95))

    def _gripper_closed(self, side: str, measured_width: float) -> bool:
        cfg: StackBowlsConfig = self.config
        teleop_target = None
        if side == "left" and self.control_mode == ControlMode.TELEOP:
            teleop_target = self.teleop_target_left_gripper
        width = measured_width if teleop_target is None else teleop_target
        return bool(width <= cfg.gripper_closed_max_width)

    @staticmethod
    def _joint_distance(current: np.ndarray, target: np.ndarray) -> float:
        return float(np.linalg.norm(current - target))

    @staticmethod
    def _exp_score(distance_sq: float, sharpness: float) -> float:
        return float(np.exp(-float(sharpness) * float(distance_sq)))

    def _joint_success(
        self,
        *,
        left_stack_dist: float,
        right_stack_dist: float,
        both_released: bool,
    ) -> bool:
        cfg: StackBowlsConfig = self.config
        if left_stack_dist > cfg.stack_joint_threshold:
            return False
        if right_stack_dist > cfg.stack_joint_threshold:
            return False
        if cfg.require_release_for_joint_success and not both_released:
            return False
        return True

    def go_to_rest(self) -> None:
        if self.config.is_dummy:
            return
        self.sdk.open_gripper()
        time.sleep(0.4)
        self._go_to_home()
