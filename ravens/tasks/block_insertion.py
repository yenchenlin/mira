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

"""Insertion Tasks."""

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils

import pybullet as p


class BlockInsertion(Task):
  """Insertion Task - Base Variant."""

  # Class variables.
  rolls = [0]
  pitchs = [0]

  @classmethod
  def get_rolls(cls):
      return cls.rolls

  @classmethod
  def get_pitchs(cls):
      return cls.pitchs

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.max_steps = 3

  def reset(self, env):
    super().reset(env)
    block_id = self.add_block(env)
    targ_pose = self.add_fixture(env)
    # self.goals.append(
    #     ([block_id], [2 * np.pi], [[0]], [targ_pose], 'pose', None, 1.))
    self.goals.append(([(block_id, (2 * np.pi, None))], np.int32([[1]]),
                       [targ_pose], False, True, 'pose', None, 1))

  def add_block(self, env):
    """Add L-shaped block."""
    size = (0.1, 0.1, 0.04)
    urdf = 'insertion/ell.urdf'
    pose = self.get_random_pose(env, size)
    return env.add_object(urdf, pose)

  def add_fixture(self, env):
    """Add L-shaped fixture to place block."""
    size = (0.1, 0.1, 0.04)
    urdf = 'insertion/fixture.urdf'
    pose = self.get_random_pose(env, size)
    env.add_object(urdf, pose, 'fixed')
    return pose


class BlockInsertionTranslation(BlockInsertion):
  """Insertion Task - Translation Variant."""

  def get_random_pose(self, env, obj_size):
    pose = super(BlockInsertionTranslation, self).get_random_pose(env, obj_size)
    pos, rot = pose
    rot = utils.eulerXYZ_to_quatXYZW((0, 0, np.pi / 2))
    return pos, rot

  # Visualization positions.
  # block_pos = (0.40, -0.15, 0.02)
  # fixture_pos = (0.65, 0.10, 0.02)


class BlockInsertionEasy(BlockInsertionTranslation):
  """Insertion Task - Easy Variant."""

  def add_block(self, env):
    """Add L-shaped block in fixed position."""
    # size = (0.1, 0.1, 0.04)
    urdf = 'insertion/ell.urdf'
    pose = ((0.5, 0, 0.02), p.getQuaternionFromEuler((0, 0, np.pi / 2)))
    return env.add_object(urdf, pose)


# class BlockInsertionSixDof(BlockInsertion):
#   """Insertion Task - 6DOF Variant."""

#   def __init__(self, *args, **kwargs):
#     super().__init__(*args, **kwargs)
#     self.sixdof = True
#     self.pos_eps = 0.02

#   def add_fixture(self, env):
#     """Add L-shaped fixture to place block."""
#     size = (0.1, 0.1, 0.04)
#     urdf = 'insertion/fixture.urdf'
#     pose = self.get_random_pose_6dof(env, size)
#     env.add_object(urdf, pose, 'fixed')
#     return pose

#   def get_random_pose_6dof(self, env, obj_size):
#     pos, rot = super(BlockInsertionSixDof, self).get_random_pose(env, obj_size)
#     z = (np.random.rand() / 10) + 0.025
#     pos = (pos[0], pos[1], obj_size[2] / 2 + z)
#     roll = (np.random.rand() - 0.5) * np.pi / 2
#     pitch = (np.random.rand() - 0.5) * np.pi / 2
#     yaw = np.random.rand() * 2 * np.pi
#     rot = utils.eulerXYZ_to_quatXYZW((roll, pitch, yaw))
#     return pos, rot

  
class BlockInsertionSixDofDiscrete(BlockInsertion):
  """Insertion Task - 6DOF Variant."""

  # Class variables.
  rolls = [-np.pi/6, -np.pi/9, -np.pi/18, 0, np.pi/18, np.pi/9, np.pi/6]
  pitchs = [-np.pi/6, -np.pi/9, -np.pi/18, 0, np.pi/18, np.pi/9, np.pi/6]

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.sixdof = True
    self.pos_eps = 0.02

  def add_fixture(self, env):
    """Add L-shaped fixture to place block."""
    size = (0.1, 0.1, 0.04)
    urdf = 'insertion/fixture.urdf'
    pose = self.get_random_pose_6dof(env, size)
    env.add_object(urdf, pose, 'fixed')
    return pose

  def get_random_pose_6dof(self, env, obj_size):
    pos, rot = super(BlockInsertionSixDofDiscrete, self).get_random_pose(env, obj_size)
    z = 0.03
    pos = (pos[0], pos[1], obj_size[2] / 2 + z)
    pitch = np.random.choice(self.pitchs)
    roll = np.random.choice(self.rolls)
    yaw = np.random.rand() * 2 * np.pi
    rot = utils.eulerXYZ_to_quatXYZW((roll, pitch, yaw))
    return pos, rot


class BlockInsertionSixDof(BlockInsertion):
  """Insertion Task - 6DOF Variant."""

  # Class variables.
  roll_bounds = (-np.pi/6, np.pi/6)
  pitch_bounds = (-np.pi/6, np.pi/6)
  n_rotations = 7
  rolls = np.linspace(roll_bounds[0], roll_bounds[1], n_rotations).tolist()
  pitchs = np.linspace(pitch_bounds[0], pitch_bounds[1], n_rotations).tolist()

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.sixdof = True
    self.pos_eps = 0.02

  def add_fixture(self, env):
    """Add L-shaped fixture to place block."""
    size = (0.1, 0.1, 0.04)
    urdf = 'insertion/fixture.urdf'
    pose = self.get_random_pose_6dof(env, size)
    env.add_object(urdf, pose, 'fixed')
    return pose

  def get_random_pose_6dof(self, env, obj_size):
    pos, rot = super(BlockInsertionSixDof, self).get_random_pose(env, obj_size)
    z = 0.03
    pos = (pos[0], pos[1], obj_size[2] / 2 + z)
    roll = np.random.rand() * (self.roll_bounds[1]-self.roll_bounds[0]) + self.roll_bounds[0]
    pitch = np.random.rand() * (self.pitch_bounds[1]-self.pitch_bounds[0]) + self.pitch_bounds[0]
    yaw = np.random.rand() * 2 * np.pi
    rot = utils.eulerXYZ_to_quatXYZW((roll, pitch, yaw))
    return pos, rot


class BlockInsertionSixDofPerspective(BlockInsertion):
  """Insertion Task - 6DOF Variant.
     It has exactly the same implementation as BlockInsertionSixDof.
     I just changed the name to make sure the data and checkpoints
     are stored differently.
  """

  # Class variables.
  roll_bounds = (-np.pi/6, np.pi/6)
  pitch_bounds = (-np.pi/6, np.pi/6)
  n_rotations = 7
  rolls = np.linspace(roll_bounds[0], roll_bounds[1], n_rotations).tolist()
  pitchs = np.linspace(pitch_bounds[0], pitch_bounds[1], n_rotations).tolist()

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.sixdof = True
    self.pos_eps = 0.02

  def add_fixture(self, env):
    """Add L-shaped fixture to place block."""
    size = (0.1, 0.1, 0.04)
    urdf = 'insertion/fixture.urdf'
    pose = self.get_random_pose_6dof(env, size)
    env.add_object(urdf, pose, 'fixed')
    return pose

  def get_random_pose_6dof(self, env, obj_size):
    pos, rot = super(BlockInsertionSixDofPerspective, self).get_random_pose(env, obj_size)
    z = 0.03
    pos = (pos[0], pos[1], obj_size[2] / 2 + z)
    roll = np.random.rand() * (self.roll_bounds[1]-self.roll_bounds[0]) + self.roll_bounds[0]
    pitch = np.random.rand() * (self.pitch_bounds[1]-self.pitch_bounds[0]) + self.pitch_bounds[0]
    yaw = np.random.rand() * 2 * np.pi
    rot = utils.eulerXYZ_to_quatXYZW((roll, pitch, yaw))
    return pos, rot


class BlockInsertionFiveDofDiscrete(BlockInsertion):
  """Insertion Task - 5DOF Variant."""

  # Class variables.
  rolls = [0]
  pitchs = [-np.pi/4, 0, np.pi/4]

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.sixdof = True
    self.pos_eps = 0.02

  def add_fixture(self, env):
    """Add L-shaped fixture to place block."""
    size = (0.1, 0.1, 0.04)
    urdf = 'insertion/fixture.urdf'
    pose = self.get_random_pose_6dof(env, size)
    env.add_object(urdf, pose, 'fixed')
    return pose

  def get_random_pose_6dof(self, env, obj_size):
    pos, rot = super(BlockInsertionFiveDofDiscrete, self).get_random_pose(env, obj_size)
    
    z = 0.03
    pos = (pos[0], pos[1], obj_size[2] / 2 + z)
    yaw = np.random.rand() * 2 * np.pi

    pitch = np.random.choice(self.pitchs)
    rot = utils.eulerXYZ_to_quatXYZW((0, pitch, yaw))
    return pos, rot


class BlockInsertionNoFixture(BlockInsertion):
  """Insertion Task - No Fixture Variant."""

  def add_fixture(self, env):
    """Add target pose to place block."""
    size = (0.1, 0.1, 0.04)
    # urdf = 'insertion/fixture.urdf'
    pose = self.get_random_pose(env, size)
    return pose

  # def reset(self, env, last_info=None):
  #   self.num_steps = 1
  #   self.goal = {'places': {}, 'steps': []}

  #   # Add L-shaped block.
  #   block_size = (0.1, 0.1, 0.04)
  #   block_urdf = 'insertion/ell.urdf'
  #   block_pose = self.get_random_pose(env, block_size)
  #   block_id = env.add_object(block_urdf, block_pose)
  #   self.goal['steps'].append({block_id: (2 * np.pi, [0])})

  #   # Add L-shaped target pose, but without actually adding it.
  #   if self.goal_cond_testing:
  #     assert last_info is not None
  #     self.goal['places'][0] = self._get_goal_info(last_info)
  #     # print('\nin insertion reset, goal: {}'.format(self.goal['places'][0]))
  #   else:
  #     hole_pose = self.get_random_pose(env, block_size)
  #     self.goal['places'][0] = hole_pose
  #     # print('\nin insertion reset, goal: {}'.format(hole_pose))

  # def _get_goal_info(self, last_info):
  #   """Used to determine the goal given the last `info` dict."""
  #   position, rotation, _ = last_info[4]  # block ID=4
  #   return (position, rotation)


class BlockInsertionSixDofOOD(BlockInsertionSixDof):
    "Block Insertion Out of Distribution"

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