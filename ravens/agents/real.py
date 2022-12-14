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

"""Real-world Agents."""

import os

import numpy as np
from ravens.models.attention import Attention
from ravens.models.transport import Transport
from ravens.models.transport_ablation import TransportPerPixelLoss
from ravens.models.transport_goal import TransportGoal
from ravens.tasks import cameras
from ravens.utils import utils
from ravens.agents.transporter import TransporterAgent
import tensorflow as tf


class RealTransporterAgent(TransporterAgent):
  """Real-world Agent that uses Transporter Networks."""

  def __init__(self, name, task, root_dir, n_rotations=36):
    super(RealTransporterAgent, self).__init__(name, task, root_dir, n_rotations=n_rotations)

    # Overwrite the bounds.
    self.bounds = np.array([[0.4, 0.7], [-0.3, 0.3], [-0.2, 0.0]])
    self.pix_size = 0.001875

    # TODO: Make the attention module consider n_rotations.
    self.attention = Attention(
        in_shape=self.in_shape,
        n_rotations=1,
        preprocess=utils.preprocess)
    self.transport = Transport(
        in_shape=self.in_shape,
        n_rotations=n_rotations,
        crop_size=self.crop_size,
        preprocess=utils.preprocess)

  def get_image(self, obs, info):
    """Stack color and height images image."""

    # if self.use_goal_image:
    #   colormap_g, heightmap_g = utils.get_fused_heightmap(goal, configs)
    #   goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
    #   input_image = np.concatenate((input_image, goal_image), axis=2)
    #   assert input_image.shape[2] == 12, input_image.shape

    # # Get color and height maps from RGB-D images.
    # cmap, hmap = utils.get_fused_heightmap(
    #     obs, self.cam_config, self.bounds, self.pix_size)

    cmap, hmap = utils.get_fused_heightmap(
        obs, [info], self.bounds, self.pix_size)

    # import matplotlib.pyplot as plt
    # plt.imsave('./tmp.png', cmap)
    # from IPython import embed; embed()
    img = np.concatenate((cmap,
                          hmap[Ellipsis, None],
                          hmap[Ellipsis, None],
                          hmap[Ellipsis, None]), axis=2)
    assert img.shape == self.in_shape, img.shape
    return img

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

    (obs, act, _, info), _ = dataset.sample()
    img = self.get_image(obs, info)
    # import matplotlib.pyplot as plt
    # from IPython import embed; embed()

    # Get training labels from data sample.
    p0_xyz, p0_xyzw = act['pose0']
    p1_xyz, p1_xyzw = act['pose1']
    p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
    p0_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
    p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)
    p1_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])

    # Data augmentation.
    if augment:
      img, _, (p0, p1), _ = utils.perturb(img, [p0, p1])

    return img, p0, p0_theta, p1, p1_theta

  def act(self, obs, info, goal=None):  # pylint: disable=unused-argument
    """Run inference and return best action given visual observations."""
    tf.keras.backend.set_learning_phase(0)

    # Get heightmap from RGB-D images.
    img = self.get_image(obs, info)

    # Attention model forward pass.
    pick_conf = self.attention.forward(img)
    argmax = np.argmax(pick_conf)
    argmax = np.unravel_index(argmax, shape=pick_conf.shape)
    p0_pix = argmax[:2]
    p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

    # Transport model forward pass.
    place_conf = self.transport.forward(img, p0_pix)
    argmax = np.argmax(place_conf)
    argmax = np.unravel_index(argmax, shape=place_conf.shape)
    p1_pix = argmax[:2]
    p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

    # Pixels to end effector poses.
    hmap = img[:, :, 3]
    p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
    p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
    p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
    p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

    return {
        'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
        'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw))
    }