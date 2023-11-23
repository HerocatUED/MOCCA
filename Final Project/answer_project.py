# 以下部分均为可更改部分，你可以把需要的数据结构定义进来
from typing import List
from bvh_loader import BVHMotion, build_loop_motion, decompose_rotation_with_yaxis
from scipy.spatial.transform import Rotation as R
from physics_warpper import PhysicsInfo
from smooth_utils import quat_to_avel, decay_spring_implicit_damping_pos, decay_spring_implicit_damping_rot
import numpy as np

DT = 30

class PDController():
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.physics_info = PhysicsInfo(viewer)
        self.cnt = 0
        self.get_pose = None
        pass
    
    def apply_pd_torque(self):
        pass

    def apply_root_force_and_torque(self):
        pass
    
    def apply_static_torque(self):
        pass

class CharacterController():
    def __init__(self, viewer, controller, pd_controller) -> None:
        # viewer 类，封装physics
        self.viewer = viewer
        # 手柄/键盘控制器
        self.controller = controller
        # pd controller
        self.pd_controller = pd_controller
        # frame_time
        self.frame_time = 0.016667
        # database
        self.database = DataBase(self.viewer.joint_name, self.controller.viewer.root_pos, self.frame_time)
        # motion
        self.motions = []
        # 添加motion
        self.motions.append(self.database.motion)
        self.motions.append(BVHMotion(bvh_file_name='./motion_material/idle_.bvh').sub_sequence(160, 160+DT))
        self.motions[1].adjust_joint_name(self.viewer.joint_name)
        self.motions[1] = build_loop_motion(self.motions[1])
        # 当前角色的参考root位置
        self.cur_root_pos = np.copy(self.controller.viewer.root_pos)
        # 当前角色的参考root旋转
        self.cur_root_rot = np.array([0,0,0,1])
        # 当前角色左右脚的相关参数
        self.foot_info = None
        
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = -1
        self.frame_cnt = 0
        self.search_frequency = DT
        self.cur_motion = self.motions[1].raw_copy()
        self.stand = True
    
    def update_state(self, 
                     desired_pos_list, 
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list
                     ):
        '''
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态
            当然我们只是提供了速度和角速度的输入，如果通过pos和rot已经很好选择下一个动作了，可以不必须使用速度和角速度
            desired_pos_list: 期望位置, 6x3的矩阵, [x, 0, z], 每一行对应0，20，40...帧的期望位置(XoZ平面)， 期望位置可以用来拟合根节点位置
            desired_rot_list: 期望旋转, 6x4的矩阵, 四元数, 每一行对应0，20，40...帧的期望旋转(Y旋转), 期望旋转可以用来拟合根节点旋转
            desired_vel_list: 期望速度, 6x3的矩阵, [x, 0, z], 每一行对应0，20，40...帧的期望速度(XoZ平面), 期望速度可以用来拟合根节点速度
            desired_avel_list: 期望角速度, 6x3的矩阵, [0, y, 0], 每一行对应0，20，40...帧的期望角速度(Y旋转), 期望角速度可以用来拟合根节点角速度
        
        Output: 输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            1. 注意应该利用的期望位置和期望速度应该都是在XoZ平面内，期望旋转和期望角速度都是绕Y轴的旋转。其他的项没有意义
        '''
        if self.cur_frame == -1:
            next_motion = None
            if np.linalg.norm(desired_vel_list[0]) < 5e-1:
                pos = self.cur_root_pos[[0,2]]
                facing_axis = R.from_quat(self.cur_root_rot).apply(np.array([0,0,1])).flatten()[[0,2]]
                next_motion = self.motions[1].translation_and_rotation(0, pos, facing_axis)
                self.stand = True
                self.foot_info = None
                self.cur_frame = 0
            else:
                self.stand = False
                self.cur_frame = self.database.search(desired_pos_list, desired_rot_list, desired_vel_list, desired_avel_list, self.foot_info)
                next_motion = self.motions[0].sub_sequence(self.cur_frame, self.cur_frame+DT)
            self.cur_motion = concatenate_two_motions(self.cur_motion, next_motion)
        else:
            self.cur_frame += 1

        joint_name = self.cur_motion.joint_name
        
        joint_transition, joint_orientation = self.cur_motion.batch_forward_kinematics(root_pos=self.cur_root_pos, frame_id_list=[self.frame_cnt])
        joint_transition = joint_transition[0]
        joint_orientation = joint_orientation[0]
        self.cur_root_rot, _ = decompose_rotation_with_yaxis(joint_orientation[0])
        
        if not self.stand:
            self.cur_root_pos += self.frame_time * self.database.root_vel[self.cur_frame]
            self.foot_info = [[joint_orientation[8], self.database.lfoot_vel[self.cur_frame], self.database.lfoot_avel[self.cur_frame]], 
                    [joint_orientation[7], self.database.rfoot_vel[self.cur_frame], self.database.rfoot_avel[self.cur_frame]]]

        self.frame_cnt += 1
        if self.frame_cnt == self.search_frequency:
            self.cur_frame = -1
            self.frame_cnt = 0
   
        return joint_name, joint_transition, joint_orientation
    

    def sync_controller_and_character(self, character_state):
        '''
        这一部分用于同步手柄和你的角色的状态
        更新后很有可能会出现手柄和角色的位置不一致，
        这里用一个简单的方案，将手柄的位置对齐于角色我位置
        '''
        controller_pos = character_state[1][0] 
        self.controller.set_pos(controller_pos)
    

class DataBase():
    def __init__(self, joint_name, init_pos, frame_time):
        print("Building Matching Map...")

        self.joint_name = joint_name
        self.frame_time = frame_time
        
        self.motion = BVHMotion(bvh_file_name='./motion_material/kinematic_motion/long_walk_.bvh').sub_sequence(180,-180)
        self.motion.adjust_joint_name(joint_name)
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/kinematic_motion/long_walk_mirror_.bvh').sub_sequence(180,-180))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/kinematic_motion/long_run_.bvh').sub_sequence(180,-180))
        self.motion.append(BVHMotion(bvh_file_name='./motion_material/kinematic_motion/long_run_mirror_.bvh').sub_sequence(180,-180))

        motion_base = self.motion.raw_copy()
        facing_axis = R.from_euler('XYZ',[0,90,0],degrees=True).apply(np.array([0,0,1])).flatten()[[0,2]]
        self.motion.append(motion_base.translation_and_rotation(0, init_pos[[0,2]], facing_axis))

        self.joint_transition, self.joint_orientation = self.motion.batch_forward_kinematics()
        # 只考虑根节点和左右脚
        self.root_transition = self.joint_transition[:,0,:]
        self.root_transition[:,1] = 0
        self.root_orientation = self.joint_orientation[:,0,:]
        for i in range(len(self.root_orientation)):
            self.root_orientation[i], _ = decompose_rotation_with_yaxis(self.root_orientation[i])
        self.root_vel = (self.root_transition[1:] - self.root_transition[:-1]) / self.frame_time
        self.root_avel = quat_to_avel(self.root_orientation, self.frame_time)
        
        self.lfoot_transition = self.joint_transition[:,8,:]
        self.lfoot_orientation = self.joint_orientation[:,8,:]
        self.lfoot_vel = (self.lfoot_transition[1:] - self.lfoot_transition[:-1]) / self.frame_time
        self.lfoot_avel = quat_to_avel(self.lfoot_orientation, self.frame_time)
        
        self.rfoot_transition = self.joint_transition[:,7,:]
        self.rfoot_orientation = self.joint_orientation[:,7,:]
        self.rfoot_vel = (self.rfoot_transition[1:] - self.rfoot_transition[:-1]) / self.frame_time
        self.rfoot_avel = quat_to_avel(self.rfoot_orientation, self.frame_time)
        
        self.foot_orientation = [self.lfoot_orientation, self.rfoot_orientation]
        self.foot_vel = [self.lfoot_vel, self.root_vel]
        self.foot_avel = [self.lfoot_avel, self.rfoot_avel]
        print(f"Done, search space: {len(self.root_vel)}")


    def search(self, desired_pos_list, desired_rot_list, desired_vel_list, desired_avel_list, foot_info, threshold_rot=0.5, threshold_cost=2):
        # future
        det_pos, future_rot, future_vel, future_avel = desired_pos_list[2]-desired_pos_list[0], desired_rot_list[2], desired_vel_list[2], desired_avel_list[2]
        cost_rot = self.root_orientation[:-1] - future_rot
        cost_rot = np.linalg.norm(cost_rot, axis=1)
        search_space = np.argwhere(cost_rot<threshold_rot).reshape(-1)
        # future
        future_cost_rot = self.root_orientation[search_space] - future_rot
        future_cost_vel = self.root_vel[search_space] - future_vel
        future_cost_avel = self.root_avel[search_space] - future_avel
        future_cost_rot = np.linalg.norm(future_cost_rot, axis=1)
        future_cost_vel = np.linalg.norm(future_cost_vel, axis=1)
        future_cost_avel = np.linalg.norm(future_cost_avel, axis=1)
        future_cost = future_cost_rot + future_cost_vel + future_cost_avel
        cost_range = np.argwhere(future_cost-np.min(future_cost)<threshold_cost).reshape(-1)
        candidate = search_space[cost_range]
        candidate -= DT
        mask = np.argwhere(candidate>=0)
        cost_range = cost_range[mask].reshape(-1)
        candidate = candidate[mask].reshape(-1)
        # now
        now_rot, now_vel, now_avel = desired_rot_list[0], desired_vel_list[0], desired_avel_list[0]
        now_cost_rot = self.root_orientation[candidate] - now_rot
        now_cost_vel = self.root_vel[candidate] - now_vel
        now_cost_avel = self.root_avel[candidate] - now_avel
        now_cost_rot = np.linalg.norm(now_cost_rot, axis=1)
        now_cost_vel = np.linalg.norm(now_cost_vel, axis=1)
        now_cost_avel = np.linalg.norm(now_cost_avel, axis=1)
        cost_pos = (self.root_transition[candidate+DT] - self.root_transition[candidate]) - det_pos
        cost_pos = np.linalg.norm(cost_pos, axis=1)
        now_cost = now_cost_rot + now_cost_vel + now_cost_avel + cost_pos
        cost = 0.3*now_cost + 0.7*future_cost[cost_range]
        # foot
        if foot_info is not None:
            cost += 0.6*self.cal_foot_cost(foot_info, candidate)
        idx = np.argwhere(cost == np.min(cost)).reshape(-1)
        return candidate[idx[0]] 

    def cal_foot_cost(self, foot_info, candidate):
        cost = np.zeros(len(candidate))
        for i in range(2):
            f_rot, f_vel, f_avel = foot_info[i][0], foot_info[i][1], foot_info[i][2]
            cost_rot = self.foot_orientation[i][candidate] - f_rot
            cost_vel = self.foot_vel[i][candidate] - f_vel
            cost_avel = self.foot_avel[i][candidate] - f_avel
            cost_rot = np.linalg.norm(cost_rot, axis=1)
            cost_vel = np.linalg.norm(cost_vel, axis=1)
            cost_avel = np.linalg.norm(cost_avel, axis=1)
            cost += cost_rot + cost_vel + cost_avel
        return cost


def concatenate_two_motions(bvh_motion1:BVHMotion, bvh_motion2:BVHMotion, frame_time:float = 0.016667, half_life:float = 0.2):
    '''
    将两个bvh动作平滑地连接起来
    输入: 将要连接的两个动作，类型为BVHMotion
          混合开始时间是第一个动作的第-1帧
    输出: 平滑地连接后的动作，类型为BVHMotion
    '''
    motion1 = bvh_motion1.raw_copy()
    motion2 = bvh_motion2.raw_copy()

    pos_diff = motion1.joint_position[-1] - motion2.joint_position[0] 
    v_diff = (motion1.joint_position[-1] - motion1.joint_position[-2])/frame_time - (motion2.joint_position[1] - motion2.joint_position[0])/frame_time 
    rot_diff = (R.from_quat(motion1.joint_rotation[-1]) * R.from_quat(motion2.joint_rotation[0]).inv()).as_rotvec()
    avel1 = quat_to_avel(motion1.joint_rotation, frame_time)
    avel2 = quat_to_avel(motion2.joint_rotation, frame_time)
    avel_diff = avel1[-1] - avel2[0]

    frame_num, joint_num, _ = np.shape(motion2.joint_position)
    for i in range(frame_num):
        for j in range(joint_num):
            pos_offset, _ = decay_spring_implicit_damping_pos(pos_diff[j], v_diff[j], half_life, i * frame_time)
            rot_offset, _ = decay_spring_implicit_damping_rot(rot_diff[j], avel_diff[j], half_life, i * frame_time)
            motion2.joint_position[i,j] += pos_offset 
            motion2.joint_rotation[i,j] = (R.from_rotvec(rot_offset) * R.from_quat(motion2.joint_rotation[i,j])).as_quat()

    return motion2