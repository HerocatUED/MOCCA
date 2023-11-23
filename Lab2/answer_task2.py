# 以下部分均为可更改部分，你可以把需要的数据结构定义进来，可以继承自Graph class
from graph import *
from answer_task1 import *
from typing import List
from bvh_motion import BVHMotion
from scipy.spatial.transform import Rotation as R

class CharacterController():
    def __init__(self, controller) -> None:
        # 手柄/键盘控制器
        self.controller = controller
        # 读取graph结构
        self.graph = Graph('./nodes.npy')
        self.graph.load_from_file()
        # node name组成的List
        self.node_names = [nd.name for nd in self.graph.nodes]
        # edge name组成的List
        self.edge_names = []
        for nd in self.graph.nodes:
            for eg in nd.edges:
                self.edge_names.append(eg.label)

        # 下面是你可能会需要的成员变量，只是一个例子形式
        # 当然，你可以任意编辑，来符合你的要求
        # 当前角色的参考root位置
        self.cur_root_pos = None
        # 当前角色的参考root旋转
        self.cur_root_rot = None
        # 当前角色处于Graph的哪一个节点
        self.cur_node : Node = None
        # 当前角色选择了哪一条边(选择要切换成哪一个新的动作)
        self.cur_edge : Edge = None
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 0
        # 当前角色对应的BVH的结束的帧数
        self.cur_end_frame = -1
        
        self.motions = None
        self.cur_motion = None
        
        # 初始化上述参数
        self.initialize()
        
    def initialize(self):
        # 当前角色处于Graph的哪一个节点
        self.cur_node = self.graph.nodes[0]
        # 当前角色选择了哪一条边(选择要切换成哪一个新的动作)
        self.cur_edge = None
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 0
        # 当前角色对应的BVH的结束的帧数
        self.cur_end_frame = self.cur_node.motion.motion_length
        
        # 当前角色的参考root位置
        self.cur_root_pos = self.cur_node.motion.joint_position[0,0,:].copy()
        self.cur_root_pos[1] = 0 # 忽略竖直方向，即y方向的位移
        
        # 当前角色的参考root旋转
        self.cur_root_rot, _ = BVHMotion.decompose_rotation_with_yaxis(self.cur_node.motion.joint_rotation[0, 0])
    
        # 抹平过渡差异
        self.motions = [BVHMotion()] * len(self.graph.nodes)
        self.motions[0] = build_loop_motion(self.graph.nodes[0].motion, 0.5, 0.2)
        for node in self.graph.nodes:
            if node.identity == 0:
                continue
            self.motions[node.identity] = smooth(node.motion, self.motions[0])

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
        if self.cur_edge is None:
            min_det = 1e+8
            best_edge = None
            # desired_rot = R.from_quat(desired_rot_list[-1]).as_rotvec()
            desired_rot = R.from_quat(desired_rot_list[-1]).as_euler('YXZ',True)
            for e in self.cur_node.edges:
                motion = copy.deepcopy(self.motions[e.destination.identity])
                pos = self.cur_root_pos[[0,2]]
                facing_axis = R.from_quat(self.cur_root_rot).apply(np.array([0,0,1])).flatten()[[0,2]]
                motion = motion.translation_and_rotation(0, pos, facing_axis)
                translation, orientation = motion.batch_forward_kinematics()
                
                next_translation = translation[-1, 0]
                next_orientation = orientation[-1, 0]
                next_translation[1] = 0
                next_orientation, _ = BVHMotion.decompose_rotation_with_yaxis(next_orientation)
                # next_orientation = R.from_quat(next_orientation[0]).as_rotvec()
                next_orientation = R.from_quat(next_orientation[0]).as_euler('YXZ',True)
                
                det_rot = abs(next_orientation - desired_rot)[0]
                det = min(det_rot, 360-det_rot)
                if det < min_det:
                    min_det = det
                    best_edge = e
                    joint_translation = translation[0]
                    joint_orientation = orientation[0]
                    joint_name = e.destination.motion.joint_name
                    self.cur_motion = motion
            
            self.cur_edge = best_edge
            self.cur_frame = 0
            self.cur_end_frame = self.cur_motion.motion_length
            self.cur_root_pos = joint_translation[0].copy()
            self.cur_root_pos[1] = 0
            self.cur_root_rot, _ = BVHMotion.decompose_rotation_with_yaxis(joint_orientation[0])

        else:
            self.cur_frame += 1        
            translation, orientation = self.cur_motion.batch_forward_kinematics()
            joint_translation = translation[self.cur_frame]
            joint_orientation = orientation[self.cur_frame]
            joint_name = self.cur_node.motion.joint_name
            self.cur_root_pos = joint_translation[0].copy()
            self.cur_root_pos[1] = 0
            self.cur_root_rot, _ = BVHMotion.decompose_rotation_with_yaxis(joint_orientation[0])

            if self.cur_frame == self.cur_end_frame-1:
                pos = self.cur_root_pos[[0,2]]
                facing_axis = R.from_quat(self.cur_root_rot).apply(np.array([0,0,1])).flatten()[[0,2]]
                self.cur_node = self.graph.nodes[0]
                self.cur_edge = None
                self.cur_motion = None

        return joint_name, joint_translation, joint_orientation  
    

def smooth(src:BVHMotion, target:BVHMotion):
    #  target --> src --> target
    res = src.raw_copy()
    # head: target --> src
    pos = target.joint_position[-1, 0, [0,2]]
    rot = target.joint_rotation[-1, 0]
    facing_axis = R.from_quat(rot).apply(np.array([0,0,1])).flatten()[[0,2]]
    res = res.translation_and_rotation(0, pos, facing_axis)

    pos_diff = target.joint_position[-1] - res.joint_position[0]
    v_diff = (target.joint_position[-1] - target.joint_position[-2])/target.frame_time - (res.joint_position[1] - res.joint_position[0])/res.frame_time
    rot_diff = (R.from_quat(target.joint_rotation[-1]) * R.from_quat(res.joint_rotation[0]).inv()).as_rotvec()
    avel2 = quat_to_avel(res.joint_rotation, res.frame_time)
    avel1 = quat_to_avel(target.joint_rotation, target.frame_time)
    avel_diff = avel1[-1] - avel2[0]
    
    half_life = 0.2

    frame_num, joint_num, _ = np.shape(res.joint_position)
    for i in range(frame_num):
        for j in range(joint_num):
            pos_offset, v_offset = decay_spring_implicit_damping_pos(pos_diff[j], v_diff[j], half_life, i * res.frame_time)
            rot_offset, avel_offset = decay_spring_implicit_damping_rot(rot_diff[j], avel_diff[j], half_life, i * res.frame_time)
            res.joint_position[i,j] += pos_offset + v_offset * res.frame_time
            res.joint_rotation[i,j] = (R.from_rotvec(rot_offset) * R.from_rotvec(avel_offset*res.frame_time) * R.from_quat(res.joint_rotation[i,j])).as_quat()

    # tail: src --> target
    pos = target.joint_position[0, 0, [0,2]]
    rot = target.joint_rotation[0, 0]
    facing_axis = R.from_quat(rot).apply(np.array([0,0,1])).flatten()[[0,2]]
    res = res.translation_and_rotation(-1, pos, facing_axis)

    pos_diff = target.joint_position[0] - res.joint_position[-1]
    v_diff = (target.joint_position[1] - target.joint_position[0])/target.frame_time - (res.joint_position[-1] - res.joint_position[-2])/res.frame_time
    rot_diff = (R.from_quat(target.joint_rotation[0]) * R.from_quat(res.joint_rotation[-1]).inv()).as_rotvec()
    avel2 = quat_to_avel(res.joint_rotation, res.frame_time)
    avel1 = quat_to_avel(target.joint_rotation, target.frame_time)
    avel_diff = avel1[0] - avel2[-1]
    
    half_life = 0.2

    frame_num, joint_num, _ = np.shape(res.joint_position)
    for i in range(frame_num):
        for j in range(joint_num):
            pos_offset, v_offset = decay_spring_implicit_damping_pos(pos_diff[j], v_diff[j], half_life, (frame_num - i) * res.frame_time)
            rot_offset, avel_offset = decay_spring_implicit_damping_rot(rot_diff[j], avel_diff[j], half_life, (frame_num - i) * res.frame_time)
            res.joint_position[i,j] += pos_offset + v_offset * res.frame_time
            res.joint_rotation[i,j] = (R.from_rotvec(rot_offset) * R.from_rotvec(avel_offset*res.frame_time) * R.from_quat(res.joint_rotation[i,j])).as_quat()

    return res
