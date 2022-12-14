"""Packing Shapes task."""

import os

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils


class PackingShapes(Task):
    """Packing Shapes base class."""

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
        self.max_steps = 1
        # self.metric = 'pose'
        # self.primitive = 'pick_place'
        self.train_set = np.arange(0, 14)
        self.test_set = np.arange(14, 20)
        self.homogeneous = False
        self.sixdof = False

    def reset(self, env):
        super().reset(env)

        # Shape Names:
        shapes = {
            0: "letter R shape",
            1: "letter A shape",
            2: "triangle",
            3: "square",
            4: "plus",
            5: "letter T shape",
            6: "diamond",
            7: "pentagon",
            8: "rectangle",
            9: "flower",
            10: "star",
            11: "circle",
            12: "letter G shape",
            13: "letter V shape",
            14: "letter E shape",
            15: "letter L shape",
            16: "ring",
            17: "hexagon",
            18: "heart",
            19: "letter M shape",
        }

        n_objects = 5
        if self.mode == 'train':
            obj_shapes = np.random.choice(self.train_set, n_objects, replace=False)
        else:
            if self.homogeneous:
                obj_shapes = [np.random.choice(self.test_set, replace=False)] * n_objects
            else:
                obj_shapes = np.random.choice(self.test_set, n_objects, replace=False)

        # Shuffle colors to avoid always picking an object of the same color
        colors = self.get_colors()
        np.random.shuffle(colors)

        # Add container box.
        # zone_size = self.get_random_size(0.1, 0.15, 0.1, 0.15, 0.05, 0.05)
        zone_size = (0.08, 0.08, 0.05)
        zone_pose = self.get_random_pose_6dof(env, zone_size)
        container_template = 'container/container-template.urdf'
        half = np.float32(zone_size) / 2
        replace = {'DIM': zone_size, 'HALF': half}
        container_urdf = self.fill_template(container_template, replace)
        env.add_object(container_urdf, zone_pose, 'fixed')
        if os.path.exists(container_urdf):
            os.remove(container_urdf)

        # Add objects.
        objects = []
        template = 'kitting/object-template.urdf'
        object_points = {}
        for i in range(n_objects):
            shape = obj_shapes[i]
            size = (0.08, 0.08, 0.02)
            pose= self.get_random_pose(env, size)
            fname = f'{shape:02d}.obj'
            fname = os.path.join(self.assets_root, 'kitting', fname)
            scale = [0.003, 0.003, 0.001]  # .0005
            replace = {'FNAME': (fname,),
                       'SCALE': scale,
                       'COLOR': colors[i]}
            urdf = self.fill_template(template, replace)
            block_id = env.add_object(urdf, pose)
            if os.path.exists(urdf):
                os.remove(urdf)
            object_points[block_id] = self.get_box_object_points(block_id)
            objects.append((block_id, (0, None)))

        # Pick the first shape.
        num_objects_to_pick = 1
        for i in range(num_objects_to_pick):
            obj_pts = dict()
            obj_pts[objects[i][0]] = object_points[objects[i][0]]

            self.goals.append(([objects[i]], np.int32([[1]]), [zone_pose],
                               False, True, 'zone',
                               (obj_pts, [(zone_pose, zone_size)]),
                               1 / num_objects_to_pick))

    def get_colors(self):
        return [
            utils.COLORS['purple'], utils.COLORS['blue'], utils.COLORS['green'],
            utils.COLORS['yellow'], utils.COLORS['red']
        ]

    def get_random_pose_6dof(self, env, obj_size):
        pos, rot = self.get_random_pose(env, obj_size)
        z = 0.035
        pos = (pos[0], pos[1], obj_size[2] / 2 + z)
        roll = np.random.rand() * (self.roll_bounds[1]-self.roll_bounds[0]) + self.roll_bounds[0]
        pitch = np.random.rand() * (self.pitch_bounds[1]-self.pitch_bounds[0]) + self.pitch_bounds[0]
        yaw = np.random.rand() * 2 * np.pi
        rot = utils.eulerXYZ_to_quatXYZW((roll, pitch, yaw))
        return pos, rot