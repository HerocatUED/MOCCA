import numpy as np
import math
from scipy.spatial.transform import Rotation as R

def align_quat(qt: np.ndarray, inplace: bool):
    ''' 
    使得给出的四元数组处于同一个半球
    '''
    qt = np.asarray(qt)
    if qt.shape[-1] != 4:
        raise ValueError('qt has to be an array of quaterions')

    if not inplace:
        qt = qt.copy()

    if qt.size == 4:  # do nothing since there is only one quation
        return qt

    sign = np.sum(qt[:-1] * qt[1:], axis=-1)
    sign[sign < 0] = -1
    sign[sign >= 0] = 1
    sign = np.cumprod(sign, axis=0, )

    qt[1:][sign < 0] *= -1
    return qt

def quat_to_avel(rot, dt):
    '''
    用有限差分计算角速度, 假设第一维度是时间
    '''
    rot = align_quat(rot, inplace=False)
    quat_diff = (rot[1:] - rot[:-1])/dt
    quat_diff[...,-1] = (1 - np.sum(quat_diff[...,:-1]**2, axis=-1)).clip(min = 0)**0.5
    quat_tmp = rot[:-1].copy()
    quat_tmp[...,:3] *= -1
    shape = quat_diff.shape[:-1]
    rot_tmp = R.from_quat(quat_tmp.reshape(-1, 4)) * R.from_quat(quat_diff.reshape(-1, 4))
    return 2 * rot_tmp.as_quat().reshape( shape + (4, ) )[...,:3]

def halflife2dampling(halflife):
    '''
    半衰期half-life到弹簧振子的damping的转换
    '''
    return 4 * math.log(2) / halflife

def decay_spring_implicit_damping_pos(pos, vel, halflife, dt):
    '''
    一个阻尼弹簧, 用来衰减位置
    '''
    d = halflife2dampling(halflife)/2
    j1 = vel + d * pos
    eydt = math.exp(-d * dt)
    pos = eydt * (pos+j1*dt)
    vel = eydt * (vel - j1 * dt * d)
    return pos, vel

def decay_spring_implicit_damping_rot(rot, avel, halflife, dt):
    '''
    一个阻尼弹簧, 用来衰减旋转
    '''
    d = halflife2dampling(halflife)/2
    j0 = rot
    j1 = avel + d * j0
    eydt = math.exp(-d * dt)
    a1 = eydt * (j0+j1*dt)
    
    rot_res = R.from_rotvec(a1).as_rotvec()
    avel_res = eydt * (avel - j1 * dt * d)
    return rot_res, avel_res

