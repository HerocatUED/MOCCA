import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
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
            motion_data.append(np.array(data).reshape(1, -1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = None
    joint_parent = None
    joint_offset = None

    #### Write Your Code Here ####
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        cnt = 0
        parent = [0]
        for i in range(len(lines)):
            if lines[i].startswith('ROOT'):
                joint_name = [lines[i][lines[i].find("ROOT")+4:].strip()]
                joint_parent = [-1]
                i += 2
                data = [float(x) for x in lines[i]
                        [lines[i].find("OFFSET")+6:].split()]
                joint_offset = [np.array(data).reshape(1, -1)]
                break
        for line in lines[i+2:]:
            if "JOINT" in line:
                joint_name.append(line[line.find("JOINT")+5:].strip())
                joint_parent.append(parent[-1])
                cnt += 1
            elif "OFFSET" in line:
                data = [float(x) for x in line[line.find("OFFSET")+6:].split()]
                joint_offset.append(np.array(data).reshape(1, -1))
            elif "{" in line:
                parent.append(cnt)
            elif "}" in line:
                parent.pop()
        joint_offset = np.concatenate(joint_offset, axis=0)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)

    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = None
    joint_orientations = None

    #### Write Your Code Here ####
    M = len(joint_name)
    motion = motion_data[frame_id]
    joint_positions = [motion[0:3]]
    joint_orientations = [R.from_euler('XYZ', motion[3:6], degrees=True)]
    motion = motion[3:]
    for i in range(1, M):
        local_rotation = R.from_euler('XYZ', motion[3*i:3*i+3], degrees=True)
        joint_orientations.append(joint_orientations[joint_parent[i]]*local_rotation)
        joint_positions.append(joint_positions[joint_parent[i]]+joint_orientations[joint_parent[i]].apply(joint_offset[i], inverse=False))
    for j in range(len(joint_orientations)):
        joint_orientations[j] = joint_orientations[j].as_quat()
    joint_positions = np.concatenate(joint_positions, axis=0).reshape((-1, 3))
    joint_orientations = np.concatenate(joint_orientations, axis=0).reshape((-1, 4))
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据

    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = None

    #### Write Your Code Here ####
    A_motion_data = load_motion_data(A_pose_bvh_path)
    A_joint_name, A_joint_parent, A_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)
    T_joint_name, T_joint_parent, T_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    A_to_T_orientations = [[0, 0, 0]]*len(T_joint_name)
    # calculate orientations from A_pose to T_pose
    for i in range(len(T_joint_name)):
        A_offset = A_joint_offset[A_joint_name.index(T_joint_name[i])]
        T_offset = T_joint_offset[i]
        A_to_T, _ = R.align_vectors([T_offset], [A_offset])
        A_to_T = A_to_T.as_euler('XYZ', degrees=True)
        A_to_T_orientations[T_joint_parent[i]] = A_to_T
    end_joints = []
    # orientations for end_joints
    for i in range(len(A_joint_name)):
        if not i in A_joint_parent:
            end_joints.append(i)
    for end_joint in end_joints:
        A_to_T_orientations[end_joint] = A_to_T_orientations[T_joint_parent[end_joint]]
    # retarget motion_data
    motion_data = np.zeros(np.shape(A_motion_data))
    motion_data[:, 0:6] = A_motion_data[:, 0:6]
    M = len(A_joint_name)
    N = np.shape(motion_data)[0]
    for n in range(N):
        for i in range(1, M):
            A_local_rotation = R.from_euler('XYZ', A_motion_data[n][3+3*i:6+3*i], degrees=True).as_matrix()
            parent_index = T_joint_name.index(A_joint_name[A_joint_parent[i]])
            joint_index = T_joint_name.index(A_joint_name[i])
            parent_A_to_T = R.from_euler('XYZ', A_to_T_orientations[parent_index], degrees=True).as_matrix()
            joint_A_to_T = np.transpose(R.from_euler('XYZ', A_to_T_orientations[joint_index], degrees=True).as_matrix())
            T_local_rotation = parent_A_to_T@A_local_rotation@joint_A_to_T
            motion_data[n][3+3*joint_index:6+3 * joint_index] = R.from_matrix(T_local_rotation).as_euler('XYZ', degrees=True)
    return motion_data
