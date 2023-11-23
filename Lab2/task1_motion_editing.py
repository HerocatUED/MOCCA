'''
在本任务中，我们会逐步实现一些动作处理的功能，包括

1. 动作的混合 
2. 动作的循环
3. 动作的拼接

'''

from viewer.controller import SimpleViewer
from answer_task1 import *
from bvh_motion import BVHMotion
import numpy as np

class ShowBVHUpdate():
    def __init__(self, viewer, joint_name, translation, orientation):
        self.viewer = viewer
        self.cur_frame = 0
        self.joint_name = joint_name
        self.translation = translation
        self.orientation = orientation
        
        
    def update(self, task):
        if not self.viewer.update_flag:
            return task.cont
        
        speed_inv = 1 # 控制播放速度的整数,越大越慢
        for i in range(len(self.joint_name)):
            self.viewer.set_joint_position_orientation(self.joint_name[i],
                                                       self.translation[self.cur_frame//speed_inv, i, :],
                                                       self.orientation[self.cur_frame//speed_inv, i, :])
        self.cur_frame = (self.cur_frame + 1) % (self.translation.shape[0]*speed_inv)
        return task.cont
    

def part1_interpolate(viewer, v):
    # 读取动作
    walk_forward = BVHMotion('motion_material/walk_forward1.bvh')
    run_forward = BVHMotion('motion_material/run_forward1.bvh')
    run_forward.adjust_joint_name(walk_forward.joint_name)
    
    # 调整方向和位置, 对齐第一帧
    # 将Root Joint挪到(0.0, 0.0)的XOZ位置上
    walk_forward = walk_forward.translation_and_rotation(0, np.array([0,0]), np.array([0,1]))
    run_forward = run_forward.translation_and_rotation(0, np.array([0,0]), np.array([0,1]))
    
    # 插值
    motion = blend_two_motions(walk_forward, run_forward, v=v, input_alpha=None)
    tanslation, orientation = motion.batch_forward_kinematics()
    task = ShowBVHUpdate(viewer, motion.joint_name, tanslation, orientation)
    viewer.addTask(task.update)
    pass

def part2_build_loop(viewer, whether_loop, ratio, half_life):
    
    motion = BVHMotion('motion_material/run_forward1.bvh')
    if whether_loop:
        motion = build_loop_motion(motion, ratio, half_life)
    
    pos = motion.joint_position[-1,0,[0,2]]
    rot = motion.joint_rotation[-1,0]
    facing_axis = R.from_quat(rot).apply(np.array([0,0,1])).flatten()[[0,2]]
    new_motion = motion.translation_and_rotation(0, pos, facing_axis)
    motion.append(new_motion)
    translation, orientation = motion.batch_forward_kinematics()
    task = ShowBVHUpdate(viewer, motion.joint_name, translation, orientation)
    viewer.addTask(task.update)


def part3_concatenate(viewer):
    mix_time = 15
    walk_forward = BVHMotion('motion_material/walk_forward1.bvh')
    walk_forward = build_loop_motion(walk_forward, 0.5, 0.2)
    walk_T = walk_forward.motion_length # 一个循环motion的周期
    
    # 由于是循环动作,把walk_forward直接拼接一遍上去,组成两段相同的动画
    motion = walk_forward
    pos = motion.joint_position[-1,0,[0,2]]
    rot = motion.joint_rotation[-1,0]
    facing_axis = R.from_quat(rot).apply(np.array([0,0,1])).flatten()[[0,2]]
    new_motion = motion.translation_and_rotation(0, pos, facing_axis)
    walk_forward.append(new_motion)
    # 手动对齐相位，取单脚落地的中间位置(single stance phase)为初始相位
    walk_phi_0 = 19
    walk_forward_T = walk_forward.sub_sequence(walk_phi_0, walk_phi_0+walk_T)
    # 为了更加明显，延长一倍长度
    motion = walk_forward_T
    pos = motion.joint_position[-1,0,[0,2]]
    rot = motion.joint_rotation[-1,0]
    facing_axis = R.from_quat(rot).apply(np.array([0,0,1])).flatten()[[0,2]]
    new_motion = motion.translation_and_rotation(0, pos, facing_axis)
    walk_forward_T.append(new_motion)


    # run_forward 类似处理
    run_forward = BVHMotion('motion_material/run_forward1.bvh')
    run_forward.adjust_joint_name(walk_forward.joint_name)
    run_forward = build_loop_motion(run_forward, 0.5, 0.2)
    run_T = run_forward.motion_length
    motion = run_forward
    pos = motion.joint_position[-1,0,[0,2]]
    rot = motion.joint_rotation[-1,0]
    facing_axis = R.from_quat(rot).apply(np.array([0,0,1])).flatten()[[0,2]]
    new_motion = motion.translation_and_rotation(0, pos, facing_axis)
    run_forward.append(new_motion)
    run_phi_0 = 28
    run_forward_T = run_forward.sub_sequence(run_phi_0, run_phi_0+run_T)
    motion = run_forward_T
    pos = motion.joint_position[-1,0,[0,2]]
    rot = motion.joint_rotation[-1,0]
    facing_axis = R.from_quat(rot).apply(np.array([0,0,1])).flatten()[[0,2]]
    new_motion = motion.translation_and_rotation(0, pos, facing_axis)
    run_forward_T.append(new_motion)
    
    # 第一次拼接
    mix_time2 = run_forward_T.motion_length - mix_time
    motion1 = concatenate_two_motions(run_forward_T, walk_forward_T, mix_time2, mix_time)

    # 第二次拼接
    mix_time3 = motion1.motion_length - 2 *mix_time
    motion2 = concatenate_two_motions(motion1, run_forward_T, mix_time3, mix_time)

    translation, orientation = motion2.batch_forward_kinematics()
    task = ShowBVHUpdate(viewer, motion2.joint_name, translation, orientation)

    viewer.addTask(task.update)
    pass


def main():
    viewer = SimpleViewer() 
    # 红色、绿色、蓝色分别代表X轴正向、Y轴正向、Z轴正向
    viewer.show_axis_frame()
    
    # 请自行取消需要的注释并更改测试setting_id
    # 请不要同时取消多个注释，否则前者会被后者覆盖
    
    # 数字代表不同期望的前进速度，因为是插值得到的
    # 请确保速度值介于[v_walk, v_run]之间
    # 也就是[1.2209,2.6177]
    # part1_interpolate(viewer, 1.5) 

    # ratio 介于[0.0, 1.0]之间
    # part2_build_loop(viewer, whether_loop=1, ratio=0.5, half_life=0.2)


    part3_concatenate(viewer) 
    viewer.run()
    
if __name__ == '__main__':
    main()