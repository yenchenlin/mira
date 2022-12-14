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

"""NeRF-porter Agent."""

import os

import numpy as np
from ravens.models.attention import Attention
from ravens.models.transport import Transport
from ravens.models.transport_ablation import TransportPerPixelLoss
from ravens.models.transport_goal import TransportGoal
from ravens.tasks import cameras
from ravens.utils import utils
from ravens.agents.transporter import OriginalTransporterAgent
import tensorflow as tf
import json
import time


class NeRFporterAgent(OriginalTransporterAgent):
  """Agent that uses NeRF-porter Networks."""

  def get_image(self, obs, name='pick_pos'):
    """Stack color and height images image."""
    cmap = obs[f'{name}_cmap']
    hmap = obs[f'{name}_hmap']

    img = np.concatenate((cmap,
                          hmap[Ellipsis, None],
                          hmap[Ellipsis, None],
                          hmap[Ellipsis, None]), axis=2)
    assert img.shape == self.in_shape, img.shape
    return img


class NeRFporter6dAgent(NeRFporterAgent):
  """Agent that uses NeRF-porter-6d Networks."""

  def __init__(self, name, task, n_rotations=36, n_input_views=36):
    super().__init__(name, task, n_rotations, n_input_views)

  def get_sample(self, dataset, augment=True):
    """Get a dataset sample.

    Args:
      dataset: a ravens.Dataset (train or validation)
      augment: if True, perform data augmentation.

    Returns:
      tuple of data for training:
        (input_image, p0, p0_theta, p1, p1_theta)
      tuple additionally includes (z, roll, pitch) if self.six_dof
      if self.use_goal_image, then the goal image is stacked with the
      current image in `input_image`. If splitting up current and goal
      images is desired, it should be done outside this method.
    """

    (obs, act, _, _), _ = dataset.sample()

    # Get training labels from data sample.
    pick_pos_img = self.get_image(obs, 'pick_pos')
    place_pos_img = self.get_image(obs, 'place_pos')
    place_pos_c2w = obs['place_pos_c2w']
    if not 'perspective'  in self.task:
      p0, p0_theta, p1, p1_theta = utils.act2pix(act, place_pos_c2w, self.bounds, self.pix_size)
    else:
      pick_pos_c2w = obs['pick_pos_c2w']
      K = np.array([
        [obs['fl_x'], 0, obs['cx']],
        [0, obs['fl_y'], obs['cy']],
        [0, 0,           1]
      ])
      p0, p0_theta, p1, p1_theta = utils.act2pix_perspective(act, pick_pos_c2w, place_pos_c2w, K)

    # Picking doesn't care about orientation.
    p1_theta = p1_theta - p0_theta
    p0_theta = 0

    # Sample negative pixels from other views.
    cmaps = obs['place_neg_cmaps']  # [B, H, W, 3]
    hmaps = obs['place_neg_hmaps']  # [B, H, W]
    place_neg_imgs = np.concatenate((cmaps,
                          hmaps[Ellipsis, None],
                          hmaps[Ellipsis, None],
                          hmaps[Ellipsis, None]), axis=3)  # [B, H, W, 6]

    # Data augmentation.
    if augment:
      # Perturb ground truth images with the same random transformation.
      pick_pos_img, place_pos_img, _, (p0, p1), _ = utils.perturb_two_images(pick_pos_img, place_pos_img, [p0, p1])

      # Also augment the negative images!
      # TODO: Remove this experimental feature which only works for batch size 1.
      assert place_neg_imgs.shape[0] == 1
      place_neg_imgs[0], _, _, _ = utils.perturb(place_neg_imgs[0], ([0, 0], [0, 0]))

    return pick_pos_img, p0, p0_theta, place_pos_img, p1, p1_theta, place_neg_imgs

  def train(self, dataset, writer=None):
    """Train on a dataset sample for 1 iteration.

    Args:
      dataset: a ravens.Dataset.
      writer: a TF summary writer (for tensorboard).
    """
    tf.keras.backend.set_learning_phase(1)

    
    # plt.imsave('tmp.png', pick_pos_img[..., :3].astype(np.uint8))
    # import matplotlib.pyplot as plt
    # from IPython import embed; embed()
    time_start = time.time()
    pick_pos_img, p0, p0_theta, place_pos_img, p1, p1_theta, place_neg_imgs = self.get_sample(dataset)

    # Get training losses.
    step = self.total_steps + 1
    if p0[0] >= 0 and p0[0] < pick_pos_img.shape[0] and p0[1] >= 0 and p0[1] < pick_pos_img.shape[1] and \
       p1[0] >= 0 and p1[0] < place_pos_img.shape[0] and p1[1] >= 0 and p1[1] < place_pos_img.shape[1]:
      loss0 = self.attention.train(pick_pos_img, p0, p0_theta)
      loss1 = self.transport.train_6d(pick_pos_img, p0, place_pos_img, p1, p1_theta, place_neg_imgs)
      with writer.as_default():
        sc = tf.summary.scalar
        sc('train_loss/attention', loss0, step)
        sc('train_loss/transport', loss1, step)
      time_spent = time.time() - time_start
      print(f'Train Iter: {step} Loss: {loss0:.4f} {loss1:.4f} Time: {time_spent:.2f} seconds')
      self.total_steps = step
    else:
      print(f'Train Iter: {step}, skip bad data; p0: {p0}, p1: {p1}; image shape: {pick_pos_img.shape}')

  def act(self, obs, info=None, goal=None):  # pylint: disable=unused-argument
    """Run inference and return best action given visual observations."""
    tf.keras.backend.set_learning_phase(0)

    # Get heightmap from RGB-D images.
    pick_pos_img = self.get_image(obs, 'pick_pos')    

    # Attention model forward pass.
    pick_conf = self.attention.forward(pick_pos_img)
    argmax = np.argmax(pick_conf)
    argmax = np.unravel_index(argmax, shape=pick_conf.shape)
    p0_pix = argmax[:2]
    p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

    # Pixels to end effector poses.
    hmap = pick_pos_img[:, :, 3]
    p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
    p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))

    # Create kernel from the pick image and the pick location.
    kernel = self.transport.get_kernel(pick_pos_img, p0_pix)

    # Enumerate place views.
    maxs = []
    argmaxs = []
    for place_img in obs['place_imgs']:
      # Transport model forward pass.
      place_conf = self.transport.forward_with_kernel(kernel, place_img, p0_pix)
      max = np.max(place_conf)
      argmax = np.argmax(place_conf)
      argmax = np.unravel_index(argmax, shape=place_conf.shape)
      maxs.append(max)
      argmaxs.append(argmax)

    # Find the view that has the max affordance.
    maxs_sorted = sorted(maxs, reverse=True)
    idx_best = maxs.index(maxs_sorted[0])

    # Use the best view's argmax to decide (x, y, yaw).
    argmax = argmaxs[idx_best]
    p1_pix = argmax[:2]
    p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

    # Use the best view's c2w to decide (roll, pitch).
    c2w = obs['c2ws'][idx_best]
    ray_dir = c2w[:3, :3] @ np.array([0, 0, 1])
    # Use the best view's height map to decide (z).
    hmap = obs['place_imgs'][idx_best][:, :, 3]
    v, u = p1_pix

    t = hmap[v, u]
    cam_center = c2w[:3, -1]
    offset_w = (np.array([u, v]) - np.array([80, 160])) * self.pix_size
    origin = cam_center + offset_w[0] * c2w[:3, :3] @ np.array([1, 0, 0]) + offset_w[1] * c2w[:3, :3] @ np.array([0, 1, 0])
    p1_xyz = origin + ray_dir * t

    c2w_eulerXYZ = utils.rotm_to_eulerXYZ(c2w[:3, :3])
    p1_xyzw = utils.eulerXYZ_to_quatXYZW((-np.pi-c2w_eulerXYZ[0], c2w_eulerXYZ[1], -p1_theta))

    return {
        'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
        'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw))
    }