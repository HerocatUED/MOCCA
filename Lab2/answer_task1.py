"""
注释里统一N表示帧数，M表示关节数
position, rotation表示局部平移和旋转
translation, orientation表示全局平移和旋转
"""
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from bvh_motion import BVHMotion
from smooth_utils import *

# part1
def blend_two_motions(bvh_motion1:BVHMotion, bvh_motion2:BVHMotion, v:float=None, input_alpha:np.ndarray=None, target_fps=60) -> BVHMotion:
    '''
    输入: 两个将要blend的动作，类型为BVHMotion
          将要生成的BVH的速度v
          如果给出插值的系数alpha就不需要再计算了
          target_fps,将要生成BVH的fps
    输出: blend两个BVH动作后的动作，类型为BVHMotion
    假设两个动作的帧数分别为n1, n2
    首先需要制作blend 的权重适量 alpha
    插值系数alpha: 0~1之间的浮点数组，形状为(n3,)
    返回的动作有n3帧，第i帧由(1-alpha[i]) * bvh_motion1[j] + alpha[i] * bvh_motion2[k]得到
    i均匀地遍历0~n3-1的同时，j和k应该均匀地遍历0~n1-1和0~n2-1
    Tips:
        1. 计算速度，两个BVH已经将Root Joint挪到(0.0, 0.0)的XOZ位置上了
        2. 利用v计算插值系数alpha
        3. 线性插值以及Slerp
        4. 可能输入的两个BVH的fps不同，需要考虑
    '''
    
    
    res = bvh_motion1.raw_copy()
    res.joint_position = np.zeros_like(res.joint_position)
    res.joint_rotation = np.zeros_like(res.joint_rotation)
    res.joint_rotation[...,3] = 1.0
    res.frame_time = 1.0 / target_fps

    # TODO: 你的代码
    n1 = np.shape(bvh_motion1.joint_position)[0]
    n2 = np.shape(bvh_motion2.joint_position)[0]

    alpha = None
    if input_alpha is None:
        # 利用Hips关节的水平位移计算运动速度
        t1 = bvh_motion1.frame_time * n1
        dx1 = bvh_motion1.joint_position[-1, 0] - bvh_motion1.joint_position[0, 0]
        dx1[1] = 0  # 计算移动速度只考虑水平位移
        s1 = np.dot(dx1, dx1) ** 0.5
        v1 = s1 / t1
        t2 = bvh_motion2.frame_time * n2
        dx2 = bvh_motion2.joint_position[-1, 0] - bvh_motion2.joint_position[0, 0]
        dx2[1] = 0  # 计算移动速度只考虑水平位移
        s2 = np.dot(dx2, dx2) ** 0.5
        v2 = s2 / t2
        # 计算权重alpha
        weight = (v - v1) / (v2 - v1)
        n3 = int(((1-weight)*v1*n1+weight*v2*n2)/v)
        alpha = np.ones(n3) * weight
    else:
        alpha = input_alpha
        n3 = np.shape(alpha)[0]
    # 调整为n3帧
    n3 = target_fps
    pos_shape = list(np.shape(bvh_motion1.joint_position))
    rot_shape = list(np.shape(bvh_motion1.joint_rotation))
    pos_shape[0] = n3
    rot_shape[0] = n3
    res.joint_position = np.zeros(pos_shape)
    res.joint_rotation = np.zeros(rot_shape)
    res.joint_rotation[..., 3] = 1.0
    # BVH帧率缩放和动作插值
    dn1 = (n1-1) / (n3-1) 
    dn2 = (n2-1) / (n3-1)
    time1 = []
    time2 = []
    for i in range(n3):
        frame1 = dn1 * i
        frame2 = dn2 * i
        joint_position1 = lerp(frame1, bvh_motion1.joint_position)
        joint_position2 = lerp(frame2, bvh_motion2.joint_position)
        res.joint_position[i] = (1 - alpha[i]) * joint_position1 + alpha[i] * joint_position2
        time1.append(frame1)
        time2.append(frame2)
    for j in range(np.shape(bvh_motion1.joint_position)[1]):
        key_rot1 = R.from_quat(bvh_motion1.joint_rotation[:,j])
        key_rot2 = R.from_quat(bvh_motion2.joint_rotation[:,j])
        slp1 = Slerp([i for i in range(n1)], key_rot1)  
        slp2 = Slerp([i for i in range(n2)], key_rot2)
        joint_rotation1 = slp1(time1).as_rotvec()
        joint_rotation2 = slp2(time2).as_rotvec()
        rot = (1 - alpha[0]) * joint_rotation1 + alpha[0] * joint_rotation2
        res.joint_rotation[:,j] = R.from_rotvec(rot).as_quat()
    # 用运动速度计算每一帧的时间
    # dx = res.joint_position[-1, 0] - res.joint_position[0, 0]
    # dx[1] = 0
    # ds = np.dot(dx, dx) ** 0.5
    # t = ds / v
    # res.frame_time = t / n3
    # res = to_target(res, target_fps)
    return res

def lerp(frame: float, joint_position: np.ndarray):
    floor = math.floor(frame)
    t = frame - floor
    n = np.shape(joint_position)[0]
    pos = (1-t)*joint_position[floor] + t*joint_position[(floor+1)%n]
    return pos

# def to_target(motion:BVHMotion, target:int):
#     res = motion.raw_copy()
#     frame_num, joint_num, _ = np.shape(motion.joint_position)
#     res.joint_position = np.zeros((target, joint_num, 3))
#     res.joint_rotation = np.zeros((target, joint_num, 4))
#     dn = (frame_num - 1) / (target - 1)
#     times = []
#     for i in range(target):
#         frame = dn*i
#         res.joint_position[i] = lerp(frame, motion.joint_position)
#         times.append(frame)
#     for j in range(np.shape(motion.joint_position)[1]):
#         key_rot = R.from_quat(motion.joint_rotation[:,j])
#         slp = Slerp([i for i in range(frame_num)], key_rot)  
#         res.joint_rotation[:,j] = slp(times).as_quat()
#     return res


# part2
def build_loop_motion(bvh_motion:BVHMotion, ratio:float, half_life:float) -> BVHMotion:
    '''
    输入: 将要loop化的动作，类型为BVHMotion
          damping在前在后的比例ratio, ratio介于[0,1]
          弹簧振子damping效果的半衰期 half_life
          如果你使用的方法不含上面两个参数，就忽视就可以了，因接口统一保留
    输出: loop化后的动作，类型为BVHMotion
    
    Tips:
        1. 计算第一帧和最后一帧的旋转差、Root Joint位置差 (不用考虑X和Z的位置差)
        2. 如果使用"inertialization"，可以利用`smooth_utils.py`的
        `quat_to_avel`函数计算对应角速度的差距，对应速度的差距请自己填写
        3. 逐帧计算Rotations和Postions的变化
        4. 注意 BVH的fps需要考虑，因为需要算对应时间
        5. 可以参考`smooth_utils.py`的注释或者 https://theorangeduck.com/page/creating-looping-animations-motion-capture
    
    '''
    res = bvh_motion.raw_copy()
    
    # TODO: 你的代码
    pos_diff = res.joint_position[-1] - res.joint_position[0]
    pos_diff [:,[0,2]] = 0
    v_diff = ((res.joint_position[-1] - res.joint_position[-2]) - (res.joint_position[1] - res.joint_position[0])) / res.frame_time
    rot_diff = (R.from_quat(res.joint_rotation[-1]) * R.from_quat(res.joint_rotation[0]).inv()).as_rotvec()
    avel = quat_to_avel(res.joint_rotation, res.frame_time)
    avel_diff = avel[-1] - avel[0]
    
    frame_num, joint_num, _ = np.shape(res.joint_position)
    for i in range(frame_num):
        for j in range(joint_num):
            pos_offset_start, v_offset_start = decay_spring_implicit_damping_pos(ratio * pos_diff[j], ratio * v_diff[j], half_life, i * res.frame_time)
            pos_offset_end, v_offset_end = decay_spring_implicit_damping_pos((1-ratio)*(-pos_diff[j]), (1-ratio)*v_diff[j], half_life, (frame_num-i)*res.frame_time)
            rot_offset_start, avel_offset_start = decay_spring_implicit_damping_rot(ratio * rot_diff[j], ratio * avel_diff[j], half_life, i * res.frame_time)
            rot_offset_end, avel_offset_end = decay_spring_implicit_damping_rot((1-ratio)*(-rot_diff[j]), (1-ratio)*avel_diff[j], half_life, (frame_num-i)*res.frame_time)
            
            pos_offset = pos_offset_start + pos_offset_end
            rot_offset = rot_offset_start + rot_offset_end
            v_offset = v_offset_start + v_offset_end
            avel_offset = avel_offset_start + avel_offset_end
            
            res.joint_position[i,j] += pos_offset + v_offset * res.frame_time
            res.joint_rotation[i,j] = (R.from_rotvec(rot_offset) * R.from_rotvec(avel_offset * res.frame_time) * R.from_quat(res.joint_rotation[i,j])).as_quat()
    
    return res



# part3
def concatenate_two_motions(bvh_motion1:BVHMotion, bvh_motion2:BVHMotion, mix_frame1:int, mix_time:int):
    '''
    将两个bvh动作平滑地连接起来
    输入: 将要连接的两个动作，类型为BVHMotion
          混合开始时间是第一个动作的第mix_frame1帧
          mix_time表示用于混合的帧数
    输出: 平滑地连接后的动作，类型为BVHMotion
    
    Tips:
        你可能需要用到BVHMotion.sub_sequence 和 BVHMotion.append
    '''
    res = bvh_motion1.raw_copy()
    
    # TODO: 你的代码
    res = res.sub_sequence(0, mix_frame1)
    pos = bvh_motion1.joint_position[mix_frame1-1, 0, [0,2]]
    rot = bvh_motion1.joint_rotation[mix_frame1-1, 0]
    facing_axis = R.from_quat(rot).apply(np.array([0,0,1])).flatten()[[0,2]]
    motion2 = bvh_motion2.translation_and_rotation(0, pos, facing_axis)

    pos_diff = res.joint_position[-1] - motion2.joint_position[0] 
    v_diff = (res.joint_position[-1] - res.joint_position[-2])/res.frame_time - (motion2.joint_position[1] - motion2.joint_position[0])/motion2.frame_time 
    rot_diff = (R.from_quat(res.joint_rotation[-1]) * R.from_quat(motion2.joint_rotation[0]).inv()).as_rotvec()
    avel1 = quat_to_avel(res.joint_rotation, res.frame_time)
    avel2 = quat_to_avel(motion2.joint_rotation, motion2.frame_time)
    avel_diff = avel1[-1] - avel2[0]
    
    half_life = 0.2
    ratio = 0.4

    frame_num, joint_num, _ = np.shape(res.joint_position)
    for i in range(frame_num):
        for j in range(joint_num):
            pos_offset, v_offset = decay_spring_implicit_damping_pos(ratio*-pos_diff[j], ratio*v_diff[j], half_life, (frame_num-i)*res.frame_time)
            rot_offset, avel_offset = decay_spring_implicit_damping_rot(ratio*-rot_diff[j], ratio*avel_diff[j], half_life, (frame_num-i)*res.frame_time)
            res.joint_position[i,j] += pos_offset + v_offset * res.frame_time
            res.joint_rotation[i,j] = (R.from_rotvec(rot_offset) * R.from_rotvec(avel_offset*res.frame_time) * R.from_quat(res.joint_rotation[i,j])).as_quat()

    frame_num, joint_num, _ = np.shape(motion2.joint_position)
    for i in range(frame_num):
        for j in range(joint_num):
            pos_offset, v_offset = decay_spring_implicit_damping_pos((1-ratio)*pos_diff[j], (1-ratio)*v_diff[j], half_life, i * motion2.frame_time)
            rot_offset, avel_offset = decay_spring_implicit_damping_rot((1-ratio)*rot_diff[j], (1-ratio)*avel_diff[j], half_life, i * motion2.frame_time)
            motion2.joint_position[i,j] += pos_offset + v_offset * motion2.frame_time
            motion2.joint_rotation[i,j] = (R.from_rotvec(rot_offset) * R.from_rotvec(avel_offset*motion2.frame_time) * R.from_quat(motion2.joint_rotation[i,j])).as_quat()

    res.append(motion2)
    res.frame_time = motion2.frame_time
    return res


