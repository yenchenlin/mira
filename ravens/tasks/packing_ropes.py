"""Packing Shapes task."""

import os

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils
from ravens.tasks import primitives

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


class PackingRopes(Task):
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
        self.primitive = primitives.PickPlace(speed=0.001)
        self.sixdof = True
        self.pos_eps = 0.05

    def reset(self, env):
        super().reset(env)

        # Add container box.
        # zone_size = self.get_random_size(0.1, 0.15, 0.1, 0.15, 0.05, 0.05)
        zone_size = (0.15, 0.15, 0.15)
        zone_pose = self.get_random_pose_6dof(env, zone_size)
        container_template = 'container/container-template.urdf'
        half = np.float32(zone_size) / 2
        replace = {'DIM': zone_size, 'HALF': half}
        container_urdf = self.fill_template(container_template, replace)
        env.add_object(container_urdf, zone_pose, 'fixed')
        if os.path.exists(container_urdf):
            os.remove(container_urdf)

        n_parts = 10
        radius = 0.005
        length = 2 * radius * n_parts * np.sqrt(2)

        # Add 3-sided square.
        square_size = (length, length, 0)
        square_pose = self.get_random_pose(env, square_size)
        # square_template = 'square/square-template.urdf'
        # replace = {'DIM': (length,), 'HALF': (length / 2 - 0.005,)}
        # urdf = self.fill_template(square_template, replace)
        # env.add_object(urdf, square_pose, 'fixed')
        # os.remove(urdf)

        # Get corner points of square.
        corner0 = (length / 2, length / 2, 0.001)
        corner1 = (-length / 2, length / 2, 0.001)
        corner0 = utils.apply(square_pose, corner0)
        corner1 = utils.apply(square_pose, corner1)

        # Add cable (series of articulated small blocks).
        increment = (np.float32(corner1) - np.float32(corner0)) / n_parts
        position, _ = self.get_random_pose(env, (0.1, 0.1, 0.1))
        position = np.float32(position)
        part_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[radius] * 3)
        part_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius * 1.5)
        parent_id = -1
        targets = []
        objects = []
        for i in range(n_parts):
            position[2] += np.linalg.norm(increment)
            part_id = p.createMultiBody(0.1, part_shape, part_visual,
                                        basePosition=position)
            if parent_id > -1:
                constraint_id = p.createConstraint(
                    parentBodyUniqueId=parent_id,
                    parentLinkIndex=-1,
                    childBodyUniqueId=part_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_POINT2POINT,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=(0, 0, np.linalg.norm(increment)),
                    childFramePosition=(0, 0, 0))
                p.changeConstraint(constraint_id, maxForce=100)
            if (i > 0) and (i < n_parts - 1):
                color = utils.COLORS['red'] + [1]
                p.changeVisualShape(part_id, -1, rgbaColor=color)
            env.obj_ids['rigid'].append(part_id)
            parent_id = part_id
            objects.append((part_id, (0, None)))
            targets.append(zone_pose)

        matches = np.clip(np.eye(n_parts) + np.eye(n_parts)[::-1], 0, 1)

        self.goals.append((objects, matches, targets,
                        False, True, 'pose', None, 1))

        for i in range(480):
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