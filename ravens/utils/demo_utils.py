import numpy as np
import json
import os
import pybullet as p
from PIL import Image
from ravens.utils import utils
from ravens.tasks.block_insertion import BlockInsertionFiveDofDiscrete, BlockInsertionSixDofDiscrete


def write_nerf_data(path, env, act, t=0.3):
  task = env.task

  rolls = task.get_rolls()
  pitchs = task.get_pitchs()
  yaw = 0
  
  pick_pos_idx = rolls.index(0) * len(pitchs) + pitchs.index(0)
  _, p1_xyzw = act['pose1']
  eulerXYZ = utils.quatXYZW_to_eulerXYZ(p1_xyzw)
  place_roll_idx = np.abs(-eulerXYZ[0] - np.array(rolls)).argmin()
  place_pitch_idx = np.abs(-eulerXYZ[1] - np.array(pitchs)).argmin()
  place_pos_idx = place_roll_idx * len(pitchs) + place_pitch_idx
    
  # Create nerf-dataset dir
  os.makedirs(path, exist_ok=True)
  image_dir = os.path.join(path, 'images')
  os.makedirs(image_dir, exist_ok=True)
  test_dir = os.path.join(path, 'test')
  os.makedirs(test_dir, exist_ok=True)

  # Define the location for cameras to look at.
  look_at = np.array([0.5, 0, 0])

  # Write test cameras.
  metadata = []
  for idx_roll in range(len(rolls)):
    for idx_pitch in range(len(pitchs)):
      idx = idx_roll * len(pitchs) + idx_pitch
      
      # Rotation.
      c2w = np.eye(4)
      c2w[2, 2] = -1
      c2w[:3, :3] = c2w[:3, :3] @ utils.eulerXYZ_to_rotm((rolls[idx_roll], pitchs[idx_pitch], yaw))
      
      # Translation.
      normal = c2w[:3, :3] @ np.array([0, 0, 1])
      ray = -1 * normal
      t = 0.3
      cam_center = look_at + ray * t
      c2w[:3, -1] = cam_center
      c2w = utils.convert_pose(c2w)

      # Convert the camera pose into OpenGL's format.
      metadata.append(
        {
          "file_path": f"./images/{idx:06}.png",
          "transform_matrix": c2w.tolist(),
        }
      )

  transforms = {}
  transforms['fl_x'] = 320.0
  transforms['fl_y'] = 320.0
  transforms['cx'] = 80.0
  transforms['cy'] = 160.0
  transforms['w'] = 160
  transforms['h'] = 320
  transforms['aabb_scale'] = 4
  transforms['scale'] = 1.0
  transforms['camera_angle_x'] = 2 * np.arctan(transforms['w'] / (2 * transforms['fl_x']))
  transforms['camera_angle_y'] = 2 * np.arctan(transforms['h'] / (2 * transforms['fl_y']))
  transforms['pick_pos_idx'] = int(pick_pos_idx)
  transforms['place_pos_idx'] = int(place_pos_idx)
  transforms['n_views'] = len(rolls) * len(pitchs)
  transforms['frames'] = metadata

  os.makedirs(path, exist_ok=True)
  with open(os.path.join(test_dir, 'transforms_test.json'), 'w') as fp:
      json.dump(transforms, fp, indent=2)

  # Write train cameras.
  metadata = []
  i = 0
  for config in env.nerf_cams:
      color, depth, _ = env.render_camera(config)
      intrinsics = np.array(config['intrinsics']).reshape(3, 3)
      position = np.array(config['position']).reshape(3, 1)
      rotation = p.getMatrixFromQuaternion(config['rotation'])
      rotation = np.array(rotation).reshape(3, 3)
      c2w = np.eye(4)
      c2w[:3, :] = np.hstack((rotation, position))

      def convert_pose(C2W):
          flip_yz = np.eye(4)
          flip_yz[1, 1] = -1
          flip_yz[2, 2] = -1
          C2W = np.matmul(C2W, flip_yz)
          return C2W

      c2w = convert_pose(c2w)
      
      Image.fromarray(color).save(os.path.join(image_dir, f"{i:06}.png"), quality=100, subsampling=0)
      # plt.imsave(os.path.join(color_dir, f"{i:06}.png"), obs['color'][i])
      # np.save(os.path.join(depth_dir, f'{i:06}.npy'), depth)
      metadata.append(
        {
            "file_path": f"./images/{i:06}.png",
            "transform_matrix": c2w.tolist(),
        }
      )

      i += 1

  transforms = {}
  transforms['fl_x'] = intrinsics[0][0]
  transforms['fl_y'] = intrinsics[1][1]
  transforms['cx'] = intrinsics[0][2]
  transforms['cy'] = intrinsics[1][2]
  transforms['w'] = config['image_size'][1]
  transforms['h'] = config['image_size'][0]
  transforms['aabb_scale'] = 4
  transforms['scale'] = 1.0
  transforms['camera_angle_x'] = 2 * np.arctan(transforms['w'] / (2 * transforms['fl_x']))
  transforms['camera_angle_y'] = 2 * np.arctan(transforms['h'] / (2 * transforms['fl_y']))
  transforms['frames'] = metadata

  with open(os.path.join(path, 'transforms.json'), 'w') as fp:
      json.dump(transforms, fp, indent=2)


def load_nerf_data(path, idx_episode, seed):
  data = {}
  def get_image(cmap, hmap):
    """Stack color and height images image."""
    img = np.concatenate((cmap,
                          hmap[Ellipsis, None],
                          hmap[Ellipsis, None],
                          hmap[Ellipsis, None]), axis=2)
    return img
  
  # Load c2ws.
  c2ws = []
  f = open (f'{path}/nerf-dataset/{idx_episode:06d}-{seed}/test/transforms_test.json', "r")
  transforms_test = json.loads(f.read())
  for frame in transforms_test['frames']:
      c2w = utils.convert_pose(np.array(frame['transform_matrix']))
      c2ws.append(c2w)
  data['c2ws'] = c2ws
  n_views_place = len(c2ws)

  # Load place images.
  place_imgs = []
  for idx_view_place in range(n_views_place):
    color_path = os.path.join(path, 'nerf-cmap', f'{idx_episode:06}-{seed}/{idx_view_place:06}.png')
    depth_path = os.path.join(path, 'nerf-hmap', f'{idx_episode:06}-{seed}/{idx_view_place:06}.npy')
    cmap = np.array(Image.open(color_path).convert('RGB'))
    hmap = np.load(depth_path)

    # Create img for transporter
    place_img = get_image(cmap, hmap)
    place_imgs.append(place_img)
  data['place_imgs'] = place_imgs
  return data