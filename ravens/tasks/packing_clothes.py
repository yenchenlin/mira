"""Packing Shapes task."""

import os

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils

import pybullet as p


def _load_softbody(basePos):
    #return p.loadSoftBody("cloth_z_up2.obj", basePosition = basePos, scale = 0.5, mass = 1., useNeoHookean = 0, useBendingSprings=1,useMassSpring=1, springElasticStiffness=40, springDampingStiffness=.1, springDampingAllDirections = 1, useSelfCollision = 0, frictionCoeff = .5, useFaceContact=1, collisionMargin = 0.04)
    return p.loadSoftBody(
        '/home/yenchenl/Workspace/nerf-porter-public/ravens/environments/assets/cloth/cloth_z_up.obj', 
        basePosition = basePos, scale = 0.1, mass = 1., useNeoHookean = 0, 
        useBendingSprings=1,useMassSpring=1, springElasticStiffness=40, 
        springDampingStiffness=.1, springDampingAllDirections = 1, 
        useSelfCollision = 0, frictionCoeff = .5, useFaceContact=1, 
        collisionMargin = 0.04)


class PackingClothes(Task):
    """Packing Clothes base class."""

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
        # self.train_set = np.arange(0, 14)
        # self.test_set = np.arange(14, 20)
        self.train_set = np.arange(0, 1)
        self.test_set = np.arange(0, 1)
        self.homogeneous = False

    def reset(self, env):
        super().reset(env)

        # Shape Names:
        # shapes = {
        #     0: "letter R shape",
        #     1: "letter A shape",
        #     2: "triangle",
        #     3: "square",
        #     4: "plus",
        #     5: "letter T shape",
        #     6: "diamond",
        #     7: "pentagon",
        #     8: "rectangle",
        #     9: "flower",
        #     10: "star",
        #     11: "circle",
        #     12: "letter G shape",
        #     13: "letter V shape",
        #     14: "letter E shape",
        #     15: "letter L shape",
        #     16: "ring",
        #     17: "hexagon",
        #     18: "heart",
        #     19: "letter M shape",
        # }
        shapes = {
            0: "black_and_blue_sneakers"
        }

        # n_objects = 5
        n_objects = 1
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
        zone_size = (0.12, 0.15, 0.15)
        zone_pose = self.get_random_pose_6dof(env, zone_size)
        container_template = 'container/container-template.urdf'
        half = np.float32(zone_size) / 2
        replace = {'DIM': zone_size, 'HALF': half}
        container_urdf = self.fill_template(container_template, replace)
        env.add_object(container_urdf, zone_pose, 'fixed')
        if os.path.exists(container_urdf):
            os.remove(container_urdf)

        # Add clothes.
        objects = []
        object_points = {}
        cloth_id = _load_softbody([0.5, 0.0, 2])
        color = [1, 0, 0, 1]
        p.changeVisualShape(cloth_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED,
                            rgbaColor=color)
            # fname = os.path.join(self.assets_root, 'google', fname)
            
            # replace = {'FNAME': (fname,),
            #            'SCALE': scale,
            #            'COLOR': colors[i]}
            # urdf = self.fill_template(template, replace)
            # block_id = env.add_object(urdf, pose)
            # if os.path.exists(urdf):
            #     os.remove(urdf)
            # object_points[block_id] = self.get_box_object_points(block_id)
            # objects.append((block_id, (0, None)))

        # # Pick the first shape.
        # num_objects_to_pick = 1
        # for i in range(num_objects_to_pick):
        #     obj_pts = dict()
        #     obj_pts[objects[i][0]] = object_points[objects[i][0]]

        #     self.goals.append(([objects[i]], np.int32([[1]]), [zone_pose],
        #                        False, True, 'zone',
        #                        (obj_pts, [(zone_pose, zone_size)]),
        #                        1 / num_objects_to_pick))

        for i in range(500):
            p.stepSimulation()


    def get_colors(self):
        return [
            utils.COLORS['purple'], utils.COLORS['blue'], utils.COLORS['green'],
            utils.COLORS['yellow'], utils.COLORS['red']
        ]

    def get_random_pose_6dof(self, env, obj_size):
        pos, rot = self.get_random_pose(env, obj_size)
        z = 0.03
        pos = (pos[0], pos[1], obj_size[2] / 2 + z)
        roll = np.random.rand() * (self.roll_bounds[1]-self.roll_bounds[0]) + self.roll_bounds[0]
        pitch = np.random.rand() * (self.pitch_bounds[1]-self.pitch_bounds[0]) + self.pitch_bounds[0]
        yaw = np.random.rand() * 2 * np.pi
        rot = utils.eulerXYZ_to_quatXYZW((roll, pitch, yaw))
        return pos, rot