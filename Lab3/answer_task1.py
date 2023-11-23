# 王想 2100013146

import numpy as np
from scipy.spatial.transform import Rotation as R
from bvh_loader import BVHMotion
from physics_warpper import PhysicsInfo


def part1_cal_torque(pose, physics_info: PhysicsInfo, **kargs):
    '''
    输入： pose: (16, 4)的numpy数组，表示每个关节的目标旋转(相对于父关节的)
           physics_info: PhysicsInfo类，包含了当前的物理信息，参见physics_warpper.py
           **kargs: 指定参数，可能包含kp,kd
    输出： global_torque: (16, 3)的numpy数组，表示每个关节的全局坐标下的目标力矩，因为不需要控制方向，根节点力矩会被后续代码无视
    '''
    # ------一些提示代码，你可以随意修改------------#
    kp = kargs.get('kp', 600) # 需要自行调整kp和kd！ 而且也可以是一个数组，指定每个关节的kp和kd
    kd = kargs.get('kd', 10) 
    parent_index = physics_info.parent_index
    joint_name = physics_info.joint_name
    # 注意关节没有自己的朝向和角速度，这里用子body的朝向和角速度表示此时关节的信息
    joint_orientation = physics_info.get_body_orientation()
    joint_avel = physics_info.get_body_angular_velocity()

    joint_rotation = (R.from_quat(joint_orientation[parent_index]).inv() * R.from_quat(joint_orientation)).as_quat()
    joint_rotation[0] = np.copy(joint_orientation[0])

    global_torque = np.zeros((16,3))

    local_torque = kp*(R.from_quat(pose)*R.from_quat(joint_rotation).inv()).as_rotvec()-kd*joint_avel
    global_torque = (R.from_quat(joint_orientation[parent_index]).apply(local_torque))
    global_torque[0] = local_torque[0]
    global_torque = np.clip(global_torque, -240, 240) 

    # # 抹去根节点力矩
    # global_torque[0] = np.zeros_like(global_torque[0])
    return global_torque

def part2_cal_float_base_torque(target_position, pose, physics_info, **kargs):
    '''
    输入： target_position: (3,)的numpy数组，表示根节点的目标位置，其余同上
    输出： global_root_force: (3,)的numpy数组，表示根节点的全局坐标下的辅助力，在后续仿真中只会保留y方向的力
           global_root_torque: (3,)的numpy数组，表示根节点的全局坐标下的辅助力矩，用来控制角色的朝向，实际上就是global_torque的第0项
           global_torque: 同上
    注意：
        1. 你需要自己计算kp和kd，并且可以通过kargs调整part1中的kp和kd
        2. global_torque[0]在track静止姿态时会被无视，但是track走路时会被加到根节点上，不然无法保持根节点朝向
        3. 可以适当将根节点目标位置上提以产生更大的辅助力，使角色走得更自然
    '''
    # ------一些提示代码，你可以随意修改------------#
    global_torque = part1_cal_torque(pose, physics_info)
    kp = kargs.get('root_kp', 6000) # 需要自行调整root的kp和kd！
    kd = kargs.get('root_kd', 10)
    root_position, root_velocity = physics_info.get_root_pos_and_vel()
    global_root_force = np.zeros((3,))
    det = np.array([0,1e-1,0])
    global_root_force = kp*(target_position+det-root_position) - kd*root_velocity
    global_root_torque = global_torque[0]
    return global_root_force, global_root_torque, global_torque

frame_cnt = 0

def part3_cal_static_standing_torque(bvh: BVHMotion, physics_info):
    '''
    输入： bvh: BVHMotion类，包含了当前的动作信息，参见bvh_loader.py
    输出： 带反馈的global_torque: (16, 3)的numpy数组，因为不需要控制方向，根节点力矩会被无视
    Tips: 
        只track第0帧就能保持站立了
        为了保持平衡可以把目标的根节点位置适当前移，比如把根节点位置和左右脚的中点加权平均，但要注意角色还会收到一个从背后推他的外力
        可以定义一个全局的frame_count变量来标记当前的帧数，在站稳后根据帧数使角色进行周期性左右摇晃
    '''
    # ------一些提示代码，你可以随意修改------------#
    tar_pos = bvh.joint_position[0][0]
    pose = bvh.joint_rotation[0]
    joint_name = physics_info.joint_name
    
    joint_positions = physics_info.get_joint_translation()
    # 适当前移，这里的权重需要你自己调整
    tar_pos = tar_pos * 0.8 + joint_positions[7] * 0.1 + joint_positions[8] * 0.1
    global frame_cnt
    # 修改根节点目标位置让角色摇摆起来，可以角色先站稳之后再周期性摇摆
    tar_pos += 0.1 * np.sin((frame_cnt)/400)
    frame_cnt += 1
    # 抹去根节点力矩
    torque = np.zeros((16,3))

    torque = part1_cal_torque(pose, physics_info)
    joint_velocity = physics_info.get_body_velocity()
    joint_mass = physics_info.get_body_mass()
    joint_mass = joint_mass.reshape((1,len(joint_mass)))
    mass = np.sum(joint_mass)
    com = joint_mass@joint_positions / mass
    com_velocity = joint_mass@joint_velocity / mass
    kp, kd = 100, 400
    virtual_force = kp * (tar_pos - com) - kd * com_velocity
    apply_list = [5,6,7,8]
    torque[apply_list] -= np.cross(com - joint_positions, virtual_force)[apply_list]

    torque[0] = np.zeros_like(torque[0])
    return torque