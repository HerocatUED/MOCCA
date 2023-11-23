# ------------- 实现一个简易的BVH对象，进行数据处理 -------------#
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from answer_task0 import load_meta_data

class BVHMotion:
    def __init__(self, bvh_file_name = None) -> None:
        # 一些 meta data
        self.joint_name = []
        self.joint_channel = []
        self.joint_parent = []
        
        # 一些local数据, 对应bvh里的channel, XYZposition和 XYZrotation
        #! 这里我们把没有XYZ position的joint的position设置为offset, 从而进行统一
        self.joint_position = None # (N,M,3) 的ndarray, 局部平移
        self.joint_rotation = None # (N,M,4)的ndarray, 用四元数表示的局部旋转
        self.frame_time = 0.016667
        
        if bvh_file_name is not None:
            self.load_motion(bvh_file_name)
        pass
    
    #------------------- 一些辅助函数 ------------------- #
    def load_motion(self, bvh_file_path):
        '''
            读取bvh文件，初始化元数据和局部数据
        '''
        self.joint_name, self.joint_parent, self.joint_channel, joint_offset = \
            self.load_meta_data(bvh_file_path)
        
        motion_data, self.frame_time = self.load_motion_data(bvh_file_path)

        # 把motion_data里的数据分配到joint_position和joint_rotation里
        self.joint_position = np.zeros((motion_data.shape[0], len(self.joint_name), 3))
        self.joint_rotation = np.zeros((motion_data.shape[0], len(self.joint_name), 4))
        self.joint_rotation[:,:,3] = 1.0 # 四元数的w分量默认为1
        
        cur_channel = 0
        for i in range(len(self.joint_name)):
            if self.joint_channel[i] == 0:
                self.joint_position[:,i,:] = joint_offset[i].reshape(1,3)
                continue   
            elif self.joint_channel[i] == 3:
                self.joint_position[:,i,:] = joint_offset[i].reshape(1,3)
                rotation = motion_data[:, cur_channel:cur_channel+3]
            elif self.joint_channel[i] == 6:
                self.joint_position[:, i, :] = motion_data[:, cur_channel:cur_channel+3]
                rotation = motion_data[:, cur_channel+3:cur_channel+6]
            self.joint_rotation[:, i, :] = R.from_euler('XYZ', rotation,degrees=True).as_quat()
            cur_channel += self.joint_channel[i]
            
        print('loaded ', bvh_file_path)
        
        return
    
    def load_meta_data(self, bvh_file_path):
        joints, joint_parents, channels, joint_offsets = load_meta_data(bvh_file_path)
        return joints, joint_parents, channels, joint_offsets

    def load_motion_data(self, bvh_file_path):
        with open(bvh_file_path, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if lines[i].startswith('Frame Time'):
                    frame_time = float(lines[i].split()[-1])
                    break
            motion_data = []
            for line in lines[i+1:]:
                data = [float(x) for x in line.split()]
                if len(data) == 0:
                    break
                motion_data.append(np.array(data).reshape(1,-1))
            motion_data = np.concatenate(motion_data, axis=0)
        return motion_data, frame_time

    def batch_forward_kinematics(self, joint_position = None, joint_rotation = None):
        '''
        利用自身的metadata进行批量前向运动学
        joint_position: (N,M,3)的ndarray, 局部平移
        joint_rotation: (N,M,4)的ndarray, 用四元数表示的局部旋转
        '''
        if joint_position is None:
            joint_position = self.joint_position
        if joint_rotation is None:
            joint_rotation = self.joint_rotation
        
        joint_translation = np.zeros_like(joint_position)
        joint_orientation = np.zeros_like(joint_rotation)
        joint_orientation[:,:,3] = 1.0 # 四元数的w分量默认为1
        
        # 一个小hack是root joint的parent是-1, 对应最后一个关节
        # 计算根节点时最后一个关节还未被计算，刚好是0偏移和单位朝向
        
        for i in range(len(self.joint_name)):
            pi = self.joint_parent[i]
            parent_orientation = R.from_quat(joint_orientation[:,pi,:]) 
            joint_translation[:, i, :] = joint_translation[:, pi, :] + \
                parent_orientation.apply(joint_position[:, i, :])
            joint_orientation[:, i, :] = (parent_orientation * R.from_quat(joint_rotation[:, i, :])).as_quat()
        return joint_translation, joint_orientation
    
    
    def adjust_joint_name(self, target_joint_name):
        '''
        调整关节顺序为target_joint_name
        '''
        idx = [self.joint_name.index(joint_name) for joint_name in target_joint_name]
        idx_inv = [target_joint_name.index(joint_name) for joint_name in self.joint_name]
        self.joint_name = [self.joint_name[i] for i in idx]
        self.joint_parent = [idx_inv[self.joint_parent[i]] for i in idx]
        self.joint_parent[0] = -1
        self.joint_channel = [self.joint_channel[i] for i in idx]
        self.joint_position = self.joint_position[:,idx,:]
        self.joint_rotation = self.joint_rotation[:,idx,:]
        pass
    
    def raw_copy(self):
        '''
        返回一个拷贝
        '''
        return copy.deepcopy(self)
    
    @property
    def motion_length(self):
        return self.joint_position.shape[0]
    
    
    def sub_sequence(self, start, end):
        '''
        返回一个子序列
        start: 开始帧
        end: 结束帧
        '''
        res = self.raw_copy()
        res.joint_position = res.joint_position[start:end,:,:]
        res.joint_rotation = res.joint_rotation[start:end,:,:]
        return res
    
    def append(self, other):
        '''
        在末尾添加另一个动作
        '''
        other = other.raw_copy()
        other.adjust_joint_name(self.joint_name)
        self.joint_position = np.concatenate((self.joint_position, other.joint_position), axis=0)
        self.joint_rotation = np.concatenate((self.joint_rotation, other.joint_rotation), axis=0)
        pass
    

    @staticmethod
    def decompose_rotation_with_yaxis(rotation):
        '''
        输入: rotation 形状为(4,)的ndarray, 四元数旋转
        输出: Ry, Rxz，分别为绕y轴的旋转和转轴在xz平面的旋转，并满足R = Ry * Rxz
        '''
        rot = R.from_quat(rotation)
        matrix = rot.as_matrix()
        yaxis = matrix[:, 1].reshape(3,)
        global_y = np.array([0,1,0]).reshape(3,)
        angle = np.arccos(np.sum(yaxis * global_y))
        axis = np.cross(yaxis, global_y)
        axis = axis / np.linalg.norm(axis)
        rot_vec = axis * angle.reshape(-1,1)
        rot_inv = R.from_rotvec(rot_vec)
        Ry = (rot_inv * rot)
        Rxz = (Ry.inv() * rot)
        Ry = Ry.as_quat()
        Rxz = Rxz.as_quat()
        return Ry, Rxz
    
    def translation_and_rotation(self, frame_num, target_translation_xz, target_facing_direction_xz):
        '''
        计算出新的joint_position和joint_rotation
        使第frame_num帧的根节点平移为target_translation_xz, 水平面朝向为target_facing_direction_xz
        frame_num: int
        target_translation_xz: (2,)的ndarray
        target_faceing_direction_xz: (2,)的ndarray，表示水平朝向。你可以理解为原本的z轴被旋转到这个方向。
        Tips:
            主要是调整root节点的joint_position和joint_rotation
            frame_num可能是负数，遵循python的索引规则
            使用decompose_rotation_with_yaxis来计算朝向轴的方位
            输入的target_facing_direction_xz的norm不一定是1
        '''
        
        res = self.raw_copy() # 拷贝一份，不要修改原始数据
        
        # 比如说，你可以这样调整第frame_num帧的根节点平移
        offset = target_translation_xz - res.joint_position[frame_num, 0, [0,2]]
        res.joint_position[:, 0, [0,2]] += offset
        Ry, Rxz = self.decompose_rotation_with_yaxis(res.joint_rotation[frame_num, 0, :])
        axis = np.array([0,1,0]).reshape(3,)
        angle = np.arctan2(target_facing_direction_xz[0], target_facing_direction_xz[1])
        desired_rot = R.from_rotvec(axis * angle)
        rot = R.from_quat(res.joint_rotation[:, 0, :])
        rot_diff = desired_rot*(R.from_quat(Ry).inv())
        res.joint_rotation[:, 0, :] = (rot_diff*rot).as_quat()
        
        pos_offset = res.joint_position[:, 0, :] - res.joint_position[frame_num, 0, :]
        rotated_pos_offset = rot_diff.apply(pos_offset)
        res.joint_position[:,0,:] = rotated_pos_offset + res.joint_position[frame_num, 0]
        return res
    

   