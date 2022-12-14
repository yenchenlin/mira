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

"""Data collection script."""

import os

from absl import app
from absl import flags

import numpy as np
import json
import pybullet as p

from ravens import tasks
from ravens.dataset import Dataset
from ravens.environments.environment import Environment
from ravens.utils.demo_utils import write_nerf_data

from PIL import Image


flags.DEFINE_string('assets_root', '.', '')
flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'towers-of-hanoi', '')
flags.DEFINE_string('mode', 'train', '')
flags.DEFINE_integer('n', 1000, '')
flags.DEFINE_bool('continuous', False, '')
flags.DEFINE_integer('steps_per_seg', 3, '')
flags.DEFINE_integer('n_input_views', 36, '')
flags.DEFINE_bool('debug', False, '')

FLAGS = flags.FLAGS


def main(unused_argv):

  # Initialize environment and task.
  env_cls = Environment
  env = env_cls(
      FLAGS.assets_root,
      disp=FLAGS.disp,
      shared_memory=FLAGS.shared_memory,
      hz=480,
      n_input_views=FLAGS.n_input_views)
  task = tasks.names[FLAGS.task](continuous=FLAGS.continuous)
  task.mode = FLAGS.mode

  # Initialize scripted oracle agent and dataset.
  agent = task.oracle(env, steps_per_seg=FLAGS.steps_per_seg)
  dataset = Dataset(os.path.join(FLAGS.data_dir, f'{FLAGS.task}-{task.mode}'))

  # Train seeds are even and test seeds are odd.
  seed = dataset.max_seed
  if seed < 0:
    seed = -1 if (task.mode == 'test') else -2
  
  # Collect training data from oracle demonstrations.
  while dataset.n_episodes < FLAGS.n:
    print(f'Oracle demonstration: {dataset.n_episodes + 1}/{FLAGS.n}')
    episode, total_reward = [], 0
    seed += 2
    np.random.seed(seed)
    env.set_task(task)
    obs, info = env.reset()
    reward = 0

    # Let agent act.
    act = agent.act(obs, info)
    episode.append((obs, act, reward, info))
    obs, reward, done, info = env.step(act)
    episode.append((obs, None, reward, info))
    print(f'Total Reward: {reward} Done: {done}')

    # Only save completed demonstrations.
    if reward > 0.99 and not FLAGS.debug:
      # Reset to the env before act.
      np.random.seed(seed)
      obs = env.reset()

      # Save NeRF's training data.
      nerf_dataset_path = f'{dataset.path}/nerf-dataset/{dataset.n_episodes:06d}-{seed}'
      if 'perspective' in FLAGS.task:
        # Use a larger t (radius of camera arrays) to see the whole scene.
        write_nerf_data(nerf_dataset_path, env, act, t=0.8)
      else:
        write_nerf_data(nerf_dataset_path, env, act)

      # Train NeRF to generate depth.
      n_steps = 5000
      os.makedirs(dataset.path, exist_ok=True)
      NGP_PATH = './orthographic-ngp'
      cmd = f'python {NGP_PATH}/scripts/run.py --mode nerf \
              --scene {nerf_dataset_path} \
              --width 160 --height 320 --n_steps {n_steps} \
              --screenshot_transforms {nerf_dataset_path}/test/transforms_test.json \
              --near_distance 0 \
              --nerfporter \
              --nerfporter_color_dir {dataset.path}/nerf-cmap/{dataset.n_episodes:06d}-{seed} \
              --nerfporter_depth_dir {dataset.path}/nerf-hmap/{dataset.n_episodes:06d}-{seed} \
              --screenshot_spp 1'
      os.system(cmd)

      # Store demonstrations.
      dataset.add(seed, episode)

if __name__ == '__main__':
  app.run(main)
