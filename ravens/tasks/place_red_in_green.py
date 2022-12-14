# coding=utf-8
# Copyright 2021 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sorting Task."""

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils

import pybullet as p


class PlaceRedInGreen(Task):
  """Sorting Task."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.max_steps = 10
    self.pos_eps = 0.05

  def reset(self, env):
    super().reset(env)
    n_bowls = np.random.randint(1, 4)
    n_blocks = np.random.randint(1, n_bowls + 1)

    # Add bowls.
    bowl_size = (0.12, 0.12, 0)
    bowl_urdf = 'bowl/bowl.urdf'
    bowl_poses = []
    for _ in range(n_bowls):
      bowl_pose = self.get_random_pose(env, bowl_size)
      env.add_object(bowl_urdf, bowl_pose, 'fixed')
      bowl_poses.append(bowl_pose)

    # Add blocks.
    blocks = []
    block_size = (0.04, 0.04, 0.04)
    block_urdf = 'stacking/block.urdf'
    for _ in range(n_blocks):
      block_pose = self.get_random_pose(env, block_size)
      block_id = env.add_object(block_urdf, block_pose)
      blocks.append((block_id, (0, None)))

    # Goal: each red block is in a different green bowl.
    self.goals.append((blocks, np.ones((len(blocks), len(bowl_poses))),
                       bowl_poses, False, True, 'pose', None, 1))

    # Colors of distractor objects.
    bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'green']
    block_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'red']

    # Add distractors.
    n_distractors = 0
    while n_distractors < 10:
      is_block = np.random.rand() > 0.5
      urdf = block_urdf if is_block else bowl_urdf
      size = block_size if is_block else bowl_size
      colors = block_colors if is_block else bowl_colors
      pose = self.get_random_pose(env, size)
      if not pose[0] or not pose[1]:
        continue
      obj_id = env.add_object(urdf, pose)
      color = colors[n_distractors % len(colors)]
      p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
      n_distractors += 1


class PlaceRedInGreenSixDofDiscrete(PlaceRedInGreen):
  """Placing Task - 6DOF Variant."""

  # Class variables.
  rolls = [-np.pi/6, 0, np.pi/6]
  pitchs = [-np.pi/6, 0, np.pi/6]

  @classmethod
  def get_rolls(cls):
      return cls.rolls

  @classmethod
  def get_pitchs(cls):
      return cls.pitchs

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.sixdof = True

  def reset(self, env):
    super(PlaceRedInGreen, self).reset(env)
    n_bowls = np.random.randint(1, 4)
    # n_blocks = np.random.randint(1, n_bowls + 1)
    n_blocks = 1

    # Add bowls.
    bowl_size = (0.12, 0.12, 0)
    bowl_urdf = 'bowl/bowl.urdf'
    bowl_poses = []
    for _ in range(n_bowls):
      bowl_pose = self.get_random_pose_6dof(env, bowl_size)
      env.add_object(bowl_urdf, bowl_pose, 'fixed')
      bowl_poses.append(bowl_pose)

    # Add blocks.
    blocks = []
    block_size = (0.04, 0.04, 0.04)
    block_urdf = 'stacking/block.urdf'
    for _ in range(n_blocks):
      block_pose = self.get_random_pose(env, block_size)
      block_id = env.add_object(block_urdf, block_pose)
      blocks.append((block_id, (0, None)))

    # Goal: each red block is in a different green bowl.
    self.goals.append((blocks, np.ones((len(blocks), len(bowl_poses))),
                       bowl_poses, False, True, 'pose', None, 1))

    # Colors of distractor objects.
    bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'green']
    block_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'red']

    # Add distractors.
    n_distractors = 0
    while n_distractors < 10:
      is_block = np.random.rand() > 0.5
      urdf = block_urdf if is_block else bowl_urdf
      size = block_size if is_block else bowl_size
      colors = block_colors if is_block else bowl_colors
      pose = self.get_random_pose(env, size)
      if not pose[0] or not pose[1]:
        continue
      obj_id = env.add_object(urdf, pose)
      color = colors[n_distractors % len(colors)]
      p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
      n_distractors += 1

  def get_random_pose_6dof(self, env, obj_size):
    pos, rot = self.get_random_pose(env, obj_size)
    z = 0.03
    pos = (pos[0], pos[1], obj_size[2] + z)
    pitch = np.random.choice(self.pitchs)
    roll = np.random.choice(self.rolls)
    yaw = np.random.rand() * 2 * np.pi
    rot = utils.eulerXYZ_to_quatXYZW((roll, pitch, yaw))
    return pos, rot


class PlaceRedInGreenSixDof(PlaceRedInGreen):
  """Placing Task - 6DOF Variant."""

  # Class variables.
  roll_bounds = (-np.pi/6, np.pi/6)
  pitch_bounds = (-np.pi/6, np.pi/6)
  n_rotations = 7
  rolls = np.linspace(roll_bounds[0], roll_bounds[1], n_rotations).tolist()
  pitchs = np.linspace(pitch_bounds[0], pitch_bounds[1], n_rotations).tolist()

  @classmethod
  def get_rolls(cls):
      return cls.rolls

  @classmethod
  def get_pitchs(cls):
      return cls.pitchs

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.sixdof = True

  def reset(self, env):
    super(PlaceRedInGreen, self).reset(env)
    n_bowls = np.random.randint(1, 4)
    # n_blocks = np.random.randint(1, n_bowls + 1)
    n_blocks = 1

    # Add bowls.
    bowl_size = (0.12, 0.12, 0)
    bowl_urdf = 'bowl/bowl.urdf'
    bowl_poses = []
    for _ in range(n_bowls):
      bowl_pose = self.get_random_pose_6dof(env, bowl_size)
      env.add_object(bowl_urdf, bowl_pose, 'fixed')
      bowl_poses.append(bowl_pose)

    # Add blocks.
    blocks = []
    block_size = (0.04, 0.04, 0.04)
    block_urdf = 'stacking/block.urdf'
    for _ in range(n_blocks):
      block_pose = self.get_random_pose(env, block_size)
      block_id = env.add_object(block_urdf, block_pose)
      blocks.append((block_id, (0, None)))

    # Goal: each red block is in a different green bowl.
    self.goals.append((blocks, np.ones((len(blocks), len(bowl_poses))),
                       bowl_poses, False, True, 'pose', None, 1))

    # Colors of distractor objects.
    bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'green']
    block_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'red']

    # Add distractors.
    n_distractors = 0
    while n_distractors < 10:
      is_block = np.random.rand() > 0.5
      urdf = block_urdf if is_block else bowl_urdf
      size = block_size if is_block else bowl_size
      colors = block_colors if is_block else bowl_colors
      pose = self.get_random_pose(env, size)
      if not pose[0] or not pose[1]:
        continue
      obj_id = env.add_object(urdf, pose)
      color = colors[n_distractors % len(colors)]
      p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
      n_distractors += 1

  def get_random_pose_6dof(self, env, obj_size):
    pos, rot = super(PlaceRedInGreenSixDof, self).get_random_pose(env, obj_size)
    z = 0.03
    pos = (pos[0], pos[1], obj_size[2] / 2 + z)
    roll = np.random.rand() * (self.roll_bounds[1]-self.roll_bounds[0]) + self.roll_bounds[0]
    pitch = np.random.rand() * (self.pitch_bounds[1]-self.pitch_bounds[0]) + self.pitch_bounds[0]
    yaw = np.random.rand() * 2 * np.pi
    rot = utils.eulerXYZ_to_quatXYZW((roll, pitch, yaw))
    return pos, rot


class PlaceRedInGreenSixDofOOD(PlaceRedInGreenSixDof):
    "Hanging Disks Out of Distribution"

    # Class variables.
    roll_bounds = (-np.pi/4, np.pi/4)
    pitch_bounds = (-np.pi/4, np.pi/4)
    n_rotations = 11
    rolls = np.linspace(roll_bounds[0], roll_bounds[1], n_rotations).tolist()
    pitchs = np.linspace(pitch_bounds[0], pitch_bounds[1], n_rotations).tolist()

    @classmethod
    def get_rolls(cls):
        return cls.rolls

    @classmethod
    def get_pitchs(cls):
        return cls.pitchs

    def get_random_pose_6dof(self, env, obj_size):
        pos, rot = self.get_random_pose(env, obj_size)
        z = 0.03
        pos = (pos[0], pos[1], obj_size[2] / 2 + z)
        roll = np.random.rand() * (self.roll_bounds[1]-self.roll_bounds[0]) + self.roll_bounds[0]
        pitch = np.random.rand() * (self.pitch_bounds[1]-self.pitch_bounds[0]) + self.pitch_bounds[0]

        ood = np.random.choice(['roll', 'pitch'])
        if ood == 'roll':
            roll = np.random.rand() * (self.roll_bounds[1]-np.pi/6) + np.pi/6
            roll = -1 * roll if np.random.rand() > 0.5 else roll 
        elif ood == 'pitch':
            pitch = np.random.rand() * (self.pitch_bounds[1]-np.pi/6) + np.pi/6
            pitch = -1 * pitch if np.random.rand() > 0.5 else pitch
            
        yaw = np.random.rand() * 2 * np.pi
        rot = utils.eulerXYZ_to_quatXYZW((roll, pitch, yaw))
        return pos, rot