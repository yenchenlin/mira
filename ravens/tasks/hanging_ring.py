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


class HangingRing(Task):
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

        # Add stand.
        base_size = (0.12, 0.36, 0.01)
        base_urdf = 'hanging_disks/stand.urdf'
        base_pose = self.get_random_pose_6dof(env, base_size)
        env.add_object(base_urdf, base_pose, 'fixed')

        # Bead properties.
        n_parts = 15
        radius = 0.005
        # The `ring_radius` (not the bead radius!) has to be tuned somewhat.
        # Try to make sure the beads don't have notable gaps between them.
        ring_radius = 0.04

        def rad_to_deg(rad):
            return (rad * 180.0) / np.pi

        def get_discretized_rotations(i, num_rotations):
            # counter-clockwise
            theta = i * (2 * np.pi) / num_rotations
            return (theta, rad_to_deg(theta))

        # Add cable (series of articulated small blocks).
        position, _ = self.get_random_pose(env, (0.1, 0.1, 0.1))
        position = np.float32(position)

        part_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[radius] * 3)
        part_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius * 1.5)
        parent_id = -1
        
        objects = []
        targets = []
        bead_positions = []
        for i in range(n_parts):
            angle_rad, _ = get_discretized_rotations(i, n_parts)
            px = ring_radius * np.cos(angle_rad)
            py = ring_radius * np.sin(angle_rad)
            bead_position = np.float32([position[0] + px, position[1] + py, 0.01])
            part_id = p.createMultiBody(0.1, part_shape, part_visual,
                                        basePosition=bead_position)
            p.changeDynamics(part_id, -1, linearDamping=10)
            if i > 0:
                if i % 5 == 0:
                    joint_type = p.JOINT_POINT2POINT
                else:
                    joint_type = p.JOINT_FIXED
                parent_frame_pos = bead_position - bead_positions[-1]
                constraint_id = p.createConstraint(
                    parentBodyUniqueId=parent_id,
                    parentLinkIndex=-1,
                    childBodyUniqueId=part_id,
                    childLinkIndex=-1,
                    jointType=joint_type,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=parent_frame_pos,
                    childFramePosition=(0, 0, 0))
                p.changeConstraint(constraint_id, maxForce=100)
                
            if i == n_parts - 1:
                joint_type = p.JOINT_FIXED
                parent_frame_pos = bead_positions[0] - bead_position
                constraint_id = p.createConstraint(
                        parentBodyUniqueId=part_id,
                        parentLinkIndex=-1,
                        childBodyUniqueId=objects[0][0],
                        childLinkIndex=-1,
                        jointType=joint_type,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=parent_frame_pos,
                        childFramePosition=(0, 0, 0))
                p.changeConstraint(constraint_id, maxForce=100)
            env.obj_ids['rigid'].append(part_id)
            parent_id = part_id
            bead_positions.append(bead_position)

            if i == 0:
                objects.append((part_id, (0, None)))
                target_pos = (base_pose[0][0] + px, base_pose[0][1] + py, base_pose[0][2])
                target_euler = utils.quatXYZW_to_eulerXYZ(base_pose[1])
                target_euler = (target_euler[0], target_euler[1], 0)
                target_rot = utils.eulerXYZ_to_quatXYZW(target_euler)
                target_pose = (target_pos, target_rot)
                targets.append(target_pose)
                

        matches = np.clip(np.eye(n_parts), 0, 1)

        self.goals.append((objects, matches, targets,
                        False, True, 'pose', None, 1))

        # # Add beaded cable. Here, `position` is the circle center.
        # position = np.float32([0.5, 0, 0.01])
        # part_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[radius]*3)
        # part_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius*1.5)

        # # Iterate through parts and create constraints as needed.
        # for i in range(n_parts):
        #     angle_rad, _ = get_discretized_rotations(n_parts)
        #     px = ring_radius * np.cos(angle_rad)
        #     py = ring_radius * np.sin(angle_rad)
        #     #print(f'pos: {px:0.2f}, {py:0.2f}, angle: {angle_rad:0.2f}, {angle_deg:0.1f}')
        #     bead_position = np.float32([position[0] + px, position[1] + py, 0.01])
        #     part_id = p.createMultiBody(0.1, part_shape, part_visual,
        #             basePosition=bead_position)
        #     color = utils.COLORS['red'] + [1]
        #     p.changeVisualShape(part_id, -1, rgbaColor=color)

        #     if i > 0:
        #         if i % 8 == 0:
        #             joint_type = p.JOINT_POINT2POINT
        #         else:
        #             joint_type = p.JOINT_FIXED
        #         parent_frame = bead_position - bead_positions_l[-1]
        #         constraint_id = p.createConstraint(
        #                 parentBodyUniqueId=beads[-1],
        #                 parentLinkIndex=-1,
        #                 childBodyUniqueId=part_id,
        #                 childLinkIndex=-1,
        #                 jointType=joint_type,
        #                 jointAxis=(0, 0, 0),
        #                 parentFramePosition=parent_frame,
        #                 childFramePosition=(0, 0, 0))
        #         p.changeConstraint(constraint_id, maxForce=100)
        #         p.changeDynamics(part_id, -1, linearDamping=10)

        #     # Make a constraint with i=0. Careful with `parent_frame`!
        #     if i == n_parts - 1:
        #         parent_frame = bead_positions_l[0] - bead_position
        #         constraint_id = p.createConstraint(
        #                 parentBodyUniqueId=part_id,
        #                 parentLinkIndex=-1,
        #                 childBodyUniqueId=beads[0],
        #                 childLinkIndex=-1,
        #                 jointType=p.JOINT_POINT2POINT,
        #                 jointAxis=(0, 0, 0),
        #                 parentFramePosition=parent_frame,
        #                 childFramePosition=(0, 0, 0))
        #         p.changeConstraint(constraint_id, maxForce=100)

        #     # Track beads.
        #     beads.append(part_id)
        #     objects.append((part_id, (0, None)))
        #     targets.append(zone_pose)
        #     bead_positions_l.append(bead_position)

        # for i in range(480):
        #     p.stepSimulation()



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