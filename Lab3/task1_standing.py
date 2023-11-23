from Viewer.viewer_new import SimpleViewer
from bvh_loader import BVHMotion
import numpy as np
np.random.seed(9527)
from answer_task1 import *
from physics_warpper import PhysicsInfo
class PDController:
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.physics_info = PhysicsInfo(viewer)
        self.cnt = 0
        self.get_pose = None
        pass
    
    def apply_pd_torque(self):
        pose = self.get_pose(self.cnt)
        torque = part1_cal_torque(pose, self.physics_info)
        torque[0] = np.zeros_like(torque[0])
        self.viewer.set_torque(torque)
        self.cnt += 1

    def apply_root_force_and_torque(self):
        position, pose = self.get_pose(self.cnt)
        global_root_force, global_root_torque, torque = part2_cal_float_base_torque(position[0], pose, self.physics_info)
        self.viewer.set_torque(torque)
        # 只保留y方向的辅助力
        global_root_force[0] = 0.0
        global_root_force[2] = 0.0
        self.viewer.set_root_force(global_root_force)
        self.viewer.set_root_torque(global_root_torque)
        self.cnt += 1
    
    def apply_static_torque(self):
        motion = self.get_pose(self.cnt)
        torque = part3_cal_static_standing_torque(motion, self.physics_info)
        torque[0] = np.zeros_like(torque[0])
        self.viewer.set_torque(torque)
        self.viewer.add_horizontal_force()
        self.cnt += 1
        
def part1_pd_control(viewer, setting=0):
    
    motion_list = [r"motion_material\idle.bvh", r"motion_material\walk.bvh"]
    motion = BVHMotion(motion_list[setting])
    # set pose
    motion.adjust_joint_name(viewer.joint_name)
    joint_translation, joint_orientation = motion.batch_forward_kinematics(frame_id_list = [0], root_pos=viewer.root_pos)
    viewer.set_pose(motion.joint_name, joint_translation[0], joint_orientation[0])

    pd_controller = PDController(viewer)
    idx_map = lambda x: (x//60)%motion.num_frames
    pd_controller.get_pose = lambda x: motion.joint_rotation[idx_map(x)]
    viewer.pre_simulation_func = pd_controller.apply_pd_torque
    pass

def part2_virtual_force(viewer, setting=0):
    motion_list = [r"motion_material\idle.bvh", r"motion_material\walk.bvh"]
    motion = BVHMotion(motion_list[setting])
    # set pose
    motion.adjust_joint_name(viewer.joint_name)
    joint_translation, joint_orientation = motion.batch_forward_kinematics(frame_id_list = [-1], root_pos=viewer.root_pos)
    viewer.set_pose(motion.joint_name, joint_translation[0], joint_orientation[0])

    pd_controller = PDController(viewer)
    idx_map = lambda x: (x//60)%motion.num_frames
    # idx_map = lambda x: 0
    pd_controller.get_pose = lambda x: (motion.joint_position[idx_map(x)], motion.joint_rotation[idx_map(x)])
    viewer.pre_simulation_func = pd_controller.apply_root_force_and_torque
    pass

def part3_static_balance(viewer, setting=0):
    viewer.add_noise_force = True
    motion_list = [r"motion_material\idle.bvh"]
    motion = BVHMotion(motion_list[setting])
    motion.adjust_joint_name(viewer.joint_name)
    joint_translation, joint_orientation = motion.batch_forward_kinematics(frame_id_list = [0], root_pos=viewer.root_pos)
    viewer.set_pose(motion.joint_name, joint_translation[0], joint_orientation[0])
    
    pd_controller = PDController(viewer)
    pd_controller.get_pose = lambda x: motion
    viewer.pre_simulation_func = pd_controller.apply_static_torque
    pass

def main():
    viewer = SimpleViewer(True, substep=32)
    # viewer.show_axis_frame()
    
    # part1_pd_control(viewer, setting=1)
    # part2_virtual_force(viewer, setting=1)
    part3_static_balance(viewer, setting=0)
    viewer.run()
    
if __name__ == '__main__':
    main()