#### delete
import os
import numpy as np
from VclSimuBackend.pymotionlib import BVHLoader
from VclSimuBackend.pymotionlib.MotionData import MotionData

def remove_toe(motion: MotionData):
    ret = motion.sub_sequence(copy=True)
    toe_idx = []
    for i, joint_name in enumerate(motion.joint_names):
        if 'Toe' in joint_name or 'Wrist' in joint_name:
            toe_idx.append(i)
    joint_idx: np.ndarray = np.arange(0, ret._num_joints, dtype=np.uint64)
    joint_idx: np.ndarray = np.delete(joint_idx, np.array(toe_idx, dtype=np.uint64))
    if ret._joint_translation is not None:
        ret._joint_translation = ret._joint_translation[:, joint_idx, :]
    if ret._joint_rotation is not None:
        ret._joint_rotation = ret._joint_rotation[:, joint_idx, :]
    if ret._joint_position is not None:
        ret._joint_position = ret._joint_position[:, joint_idx, :]
    if ret._joint_orientation is not None:
        ret._joint_orientation = ret._joint_orientation[:, joint_idx, :]
    # modify parent index
    for i in range(len(toe_idx)):
        end_idx = toe_idx[i] - i
        before = ret._skeleton_joint_parents[:end_idx]
        after = ret._skeleton_joint_parents[end_idx + 1:]
        for j in range(len(after)):
            if after[j] > end_idx:
                after[j] -= 1
        ret._skeleton_joint_parents = before + after
    # modify other attributes
    ret._skeleton_joints = np.array(ret._skeleton_joints)[joint_idx].tolist()
    ret._skeleton_joint_offsets = np.array(ret._skeleton_joint_offsets)[joint_idx]
    ret._num_joints -= len(toe_idx)
    ret._end_sites.clear()
    for end_idx in motion._end_sites:
        toe_cnt = 0
        for toe in toe_idx:
            if end_idx == toe:
                toe_cnt = -1
                continue
            if end_idx > toe:
                toe_cnt += 1
        if toe_cnt >= 0:
            ret._end_sites.append(end_idx - toe_cnt)
    return ret

if __name__ == "__main__":
    fname = './physics_motion/long_run.bvh'
    abs_fname = os.path.join(os.path.dirname(__file__), fname)
    origin_motion = BVHLoader.load(abs_fname)
    modified_motion = remove_toe(origin_motion)
    # modified_motion._joint_rotation = np.zeros_like(modified_motion._joint_rotation)
    # modified_motion._joint_rotation[:, :, 3] = 1.
    BVHLoader.save(modified_motion, os.path.join(os.path.dirname(__file__), './physics_motion/long_run.bvh' +'_.bvh'))

### delete