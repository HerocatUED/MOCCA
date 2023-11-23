import numpy as np
from scipy.spatial.transform import Rotation as R

def load_skeleton(bvh_file_path):
    """请将lab1中的part1_calculate_T_pose复制过来
    Tips:
        joint_name顺序应该和bvh一致
    """
    
    with open(bvh_file_path, 'r') as f:
        channels = []
        joints = []
        joint_parents = []
        joint_offsets = []
        end_sites = []

        parent_stack = [None]
        for line in f:
            if 'ROOT' in line or 'JOINT' in line:
                joints.append(line.split()[-1])
                joint_parents.append(parent_stack[-1])
                channels.append(['', ''])
                joint_offsets.append([0, 0, 0])

            elif 'End Site' in line:
                end_sites.append(len(joints))
                joints.append(parent_stack[-1] + '_end')
                joint_parents.append(parent_stack[-1])
                channels.append(['', ''])
                joint_offsets.append([0, 0, 0])

            elif '{' in line:
                parent_stack.append(joints[-1])

            elif '}' in line:
                parent_stack.pop()

            elif 'OFFSET' in line:
                joint_offsets[-1] = [float(x) for x in line.split()[-3:]]

            elif 'CHANNELS' in line:
                trans_order = []
                rot_order = []
                for token in line.split():
                    if 'position' in token:
                        trans_order.append(token[0])

                    if 'rotation' in token:
                        rot_order.append(token[0])

                channels[-1] = [''.join(trans_order), ''.join(rot_order)]

            elif 'Frame Time:' in line:
                break
    joint_parents = [-1]+ [joints.index(i) for i in joint_parents[1:]]
    return joints, joint_parents, joint_offsets


def load_motion_data(bvh_file_path):
    """同Lab1， 帮你读取bvh文件中的运动数据"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data

def forward_kinematics(joint_name, joint_parent, joint_offset, motion_data):
    """将lab1中的part2_forward_kinematics复制过来
    """
    num = len(joint_name)
    joint_positions = np.zeros((motion_data.shape[0], num, 3))
    joint_orientations = np.zeros((motion_data.shape[0], num, 4))
    j = 0
    for i in range(num):
        if i == 0:
            joint_positions[:, i] = joint_offset[None, i] + motion_data[:, j:j+3]
            joint_orientations[:, i] = R.from_euler('XYZ', motion_data[:, j+3:j+6], degrees=True).as_quat()
            j += 6
        else:
            parent_rot = R.from_quat(joint_orientations[:, joint_parent[i]])
            joint_positions[:, i] = joint_positions[:, joint_parent[i]] + parent_rot.apply(joint_offset[None,i])
            if 'end' not in joint_name[i]:
                joint_orientations[:, i] = (parent_rot*R.from_euler('XYZ', motion_data[:, j:j+3], degrees=True)).as_quat()
                j += 3
            
    return joint_positions, joint_orientations

# ---delete--- #
def forward_kinematics_with_channel(joint_parent, joint_channel, joint_offset, motion_data):
    num = len(joint_parent)
    joint_positions = np.zeros((motion_data.shape[0], num, 3))
    joint_orientations = np.zeros((motion_data.shape[0], num, 4))
    joint_orientations[...,3] = 1
    
    j = 0
    for i in range(num):
        
        if joint_channel[i] == 6:
            joint_positions[:, i] = joint_offset[i] + motion_data[:, j:j+3]
            rot = R.from_euler('XYZ', motion_data[:, j+3:j+6], degrees=True)
            if joint_parent[i] != -1:
                joint_positions[:, i] += joint_positions[:, joint_parent[i]]
                rot = R.from_quat(joint_orientations[:, joint_parent[i]])*rot
            joint_orientations[:, i] = rot.as_quat()
            j += 6
        else:
            parent_rot = R.from_quat(joint_orientations[:, joint_parent[i]])
            joint_positions[:, i] = joint_positions[:, joint_parent[i]] + parent_rot.apply(joint_offset[i])
            if joint_channel[i] == 3:
                joint_orientations[:, i] = (parent_rot*R.from_euler('XYZ', motion_data[:, j:j+3], degrees=True)).as_quat()
                j += 3
    return joint_positions, joint_orientations