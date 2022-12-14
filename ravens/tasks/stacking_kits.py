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

"""Kitting Tasks."""

import os

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils


class StackingKits(Task):
  """Kitting Tasks base class."""

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
    # self.ee = 'suction'
    self.max_steps = 10
    # self.metric = 'pose'
    # self.primitive = 'pick_place'
    self.train_set = np.arange(0, 14)
    # self.test_set = np.arange(14, 20)
    self.test_set = [14, 15, 19]
    self.homogeneous = False
    self.sixdof = True
    self.match_rp = True

  def reset(self, env):
    super().reset(env)

    # Add kit.
    kit_size = (0.28, 0.2, 0.005)
    kit_urdf = 'kitting/kit_sixdof.urdf'
    kit_pose = self.get_random_pose_6dof(env, kit_size)
    env.add_object(kit_urdf, kit_pose, 'fixed')

    n_objects = 1
    if self.mode == 'train':
      obj_shapes = np.random.choice(self.train_set, n_objects)
    else:
      if self.homogeneous:
        obj_shapes = [np.random.choice(self.test_set)] * n_objects
      else:
        obj_shapes = np.random.choice(self.test_set, n_objects)

    colors = [
        utils.COLORS['purple'], utils.COLORS['blue'], utils.COLORS['green'],
        utils.COLORS['yellow'], utils.COLORS['red']
    ]
    # Shuffle the colors.
    np.random.shuffle(colors)

    symmetry = [
        2 * np.pi, 2 * np.pi, 2 * np.pi / 3, np.pi / 2, np.pi / 2, 2 * np.pi,
        np.pi, 2 * np.pi / 5, np.pi, np.pi / 2, 2 * np.pi / 5, 0, 2 * np.pi,
        2 * np.pi, 2 * np.pi, 2 * np.pi, 0, 2 * np.pi / 6, 2 * np.pi, 2 * np.pi
    ]

    # Build kit.
    targets = []
    targ_pos = [[0, 0, 0.0014]]
    template = 'kitting/object-template.urdf'
    kit_euler = utils.quatXYZW_to_eulerXYZ(kit_pose[1])
    for i in range(n_objects):
      shape = os.path.join(self.assets_root, 'kitting',
                           f'{obj_shapes[i]:02d}.obj')
      scale = [0.003, 0.003, 0.001]  # .0005
      pos = utils.apply(kit_pose, targ_pos[i])
      theta = np.random.rand() * 2 * np.pi
      rot = utils.eulerXYZ_to_quatXYZW((kit_euler[0], kit_euler[1], theta))
      replace = {'FNAME': (shape,), 'SCALE': scale, 'COLOR': (0.2, 0.2, 0.2)}
      urdf = self.fill_template(template, replace)
      env.add_object(urdf, (pos, rot), 'fixed')
      os.remove(urdf)
      targets.append((pos, rot))

    # Add objects.
    objects = []
    matches = []
    # objects, syms, matcheses = [], [], []
    for i in range(n_objects):
      shape = obj_shapes[i]
      size = (0.08, 0.08, 0.02)
      pose = self.get_random_pose(env, size)
      fname = f'{shape:02d}.obj'
      fname = os.path.join(self.assets_root, 'kitting', fname)
      scale = [0.003, 0.003, 0.001]  # .0005
      replace = {'FNAME': (fname,), 'SCALE': scale, 'COLOR': colors[i]}
      urdf = self.fill_template(template, replace)
      block_id = env.add_object(urdf, pose)
      os.remove(urdf)
      objects.append((block_id, (symmetry[shape], None)))
      # objects[block_id] = symmetry[shape]
      match = np.zeros(len(targets))
      match[np.argwhere(obj_shapes == shape).reshape(-1)] = 1
      matches.append(match)
      # print(targets)
      # exit()
      # matches.append(list(np.argwhere(obj_shapes == shape).reshape(-1)))
    matches = np.int32(matches)
    # print(matcheses)
    # exit()

    # Add goal.
    # self.goals.append((objects, syms, targets, 'matches', 'pose', 1.))

    # Goal: objects are placed in their respective kit locations.
    # print(objects)
    # print(matches)
    # print(targets)
    # exit()
    self.goals.append((objects, matches, targets, False, True, 'pose', None, 1))
    # goal = Goal(objects, syms, targets)
    # metric = Metric('pose-matches', None, 1.)
    # self.goals.append((goal, metric))

    # # Goal: box is aligned with corner (1 of 4 possible poses).

  def get_random_pose_6dof(self, env, obj_size):
    pos, rot = super(StackingKits, self).get_random_pose(env, obj_size)
    z = 0.03
    pos = (pos[0], pos[1], obj_size[2] / 2 + z)
    roll = np.random.rand() * (self.roll_bounds[1]-self.roll_bounds[0]) + self.roll_bounds[0]
    pitch = np.random.rand() * (self.pitch_bounds[1]-self.pitch_bounds[0]) + self.pitch_bounds[0]
    yaw = np.random.rand() * 2 * np.pi
    rot = utils.eulerXYZ_to_quatXYZW((roll, pitch, yaw))
    return pos, rot


class StackingKitsEasy(StackingKits):
  """Kitting Task - Easy variant."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.rot_eps = np.deg2rad(30)
    self.train_set = np.int32(
        [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19])
    self.test_set = np.int32([3, 11])
    self.homogeneous = True


class StackingKitsOOD(StackingKits):
  """Kitting Task - Easy variant."""

  # Class variables.
  roll_bounds = (-np.pi/4, np.pi/4)
  pitch_bounds = (-np.pi/4, np.pi/4)
  n_rotations = 11
  rolls = np.linspace(roll_bounds[0], roll_bounds[1], n_rotations).tolist()
  pitchs = np.linspace(pitch_bounds[0], pitch_bounds[1], n_rotations).tolist()

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