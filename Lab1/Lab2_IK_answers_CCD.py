import numpy as np
from scipy.spatial.transform import Rotation as R


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """

    #### Write Your Code Here ####
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    # joints offset
    init_offset = [[0, 0, 0]]*len(meta_data.joint_name)
    joint_length = [0]*len(meta_data.joint_name)
    for i in range(1, len(meta_data.joint_name)):
        p = meta_data.joint_parent[i]
        init_offset[i] = meta_data.joint_initial_position[i] - \
            meta_data.joint_initial_position[p]
        joint_length[i] = init_offset[i].dot(init_offset[i])**0.5
    # hyperparameters
    max_iteration = 50
    error_threshold = 0.01
    num_joints = len(path)
    # CCD
    joint_local_rotation=np.zeros((len(meta_data.joint_name),4))
    joint_local_rotation[:,3]=1
    for m in range(max_iteration):
        error = joint_positions[path[-1]]-target_pose
        error = error.dot(error)**0.5
        if error < error_threshold or m > max_iteration:
            break
        for i in range(num_joints-2,-1,-1):
            to_end=joint_positions[path[-1]]-joint_positions[path[i]]
            to_target=target_pose-joint_positions[path[i]]
            rotation,_=R.align_vectors([to_target],[to_end])
            rotation=rotation.as_quat()
            joint_local_rotation[path[i]]=rotation*joint_local_rotation[path[i]]
        # forward IK
        for i in range(1,len(meta_data.joint_name)):
            p=meta_data.joint_parent[i]
            joint_orientations[i]=joint_orientations[p]*joint_local_rotation[i]
            joint_positions[i]=joint_positions[p]+R.from_quat(joint_orientations[p]).apply(init_offset[i],inverse=False)
    return joint_positions, joint_orientations


def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_height_ratio):
    """
    输入RightFoot，相较于bvh动作目标高度的比例，IK以外的部分与bvh一致
    """

    #### Write Your Code Here ####
    return joint_positions, joint_orientations
