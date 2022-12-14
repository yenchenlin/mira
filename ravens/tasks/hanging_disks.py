"""Towers of Hanoi task."""

import os
import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils


class HangingDisks(Task):
    """Towers of Hanoi task."""

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
        self.max_steps = 1
        self.match_rp = True
        self.sixdof = True

    def reset(self, env):
        super().reset(env)

        # Add stand.
        base_size = (0.12, 0.36, 0.01)
        base_urdf = 'hanging_disks/stand.urdf'
        base_pose = self.get_random_pose_6dof(env, base_size)
        env.add_object(base_urdf, base_pose, 'fixed')

        # Rod positions in base coordinates.
        rod_pos = (0, 0, 0.03)

        # Shuffle colors to avoid always picking an object of the same color
        color_names = self.get_colors()
        colors = [utils.COLORS[cn] for cn in color_names]
        np.random.shuffle(colors)

        # Add disks.
        disks = []
        n_disks = 1
        size = (0.08, 0.08, 0.02)
        pos, _ = self.get_random_pose(env, size)
        template = 'hanging_disks/disk.urdf'
        for i in range(n_disks):
            replace = {
                'FNAME': [os.path.join(self.assets_root, 'hanging_disks/disk.obj')],
                'COLOR': colors[i]
            }
            disk_urdf = self.fill_template(template, replace)
            # pos = utils.apply(base_pose, rod_pos[0])
            # z = 0.015 * (n_disks - i - 2)
            # pos = (pos[0], pos[1], pos[2] + z)
            disks.append(env.add_object(disk_urdf, (pos, (0, 0, 1, 0))))

        # Solve Hanoi sequence with dynamic programming.
        hanoi_steps = []  # [[object index, from rod, to rod], ...]

        def solve_hanoi(n, t0, t1, t2):
            if n == 0:
                hanoi_steps.append([n, t0, t1])
                return
            solve_hanoi(n - 1, t0, t2, t1)
            hanoi_steps.append([n, t0, t1])
            solve_hanoi(n - 1, t2, t1, t0)

        solve_hanoi(n_disks - 1, 0, 2, 1)

        # Goal: pick and place disks using Hanoi sequence.
        for step in hanoi_steps:
            disk_id = disks[step[0]]
            targ_pos = rod_pos
            targ_pos = utils.apply(base_pose, targ_pos)
            targ_pose = (targ_pos, base_pose[1])
            self.goals.append(([(disk_id, (0, None))], np.int32([[1]]), [targ_pose],
                               False, True, 'pose', None, 1))

    def get_random_pose_6dof(self, env, obj_size):
        pos, rot = self.get_random_pose(env, obj_size)
        z = 0.01
        pos = (pos[0], pos[1], obj_size[2] / 2 + z)
        roll = np.random.rand() * (self.roll_bounds[1]-self.roll_bounds[0]) + self.roll_bounds[0]
        pitch = np.random.rand() * (self.pitch_bounds[1]-self.pitch_bounds[0]) + self.pitch_bounds[0]
        yaw = np.random.rand() * 2 * np.pi
        rot = utils.eulerXYZ_to_quatXYZW((roll, pitch, yaw))
        return pos, rot

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS



class HangingDisksOOD(HangingDisks):
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
        z = 0.01
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