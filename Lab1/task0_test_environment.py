from viewer import SimpleViewer
from scipy.spatial.transform import Rotation as R
import numpy as np

def part0(viewer):
    """
    part0 读取角色，可以通过鼠标左右中键操作相机视角
    """
    viewer.run()

def part1(viewer):
    """
    part1 给定一个角色，和一个全局的旋转（Euler角格式），请使用Scipy-Rotation把它转换成quaternion
    然后使用show_pose来体现角色位形
    可以感受一下'XYZ'格式的Euler角和'xyz', 'ZYX', 'zyx'的欧拉角转换成四元数的区别
    请分别取消对应欧拉角格式的注释，以体现区别
    """

    info = np.load('one_piece_pose.npy', allow_pickle=True).item()
    joint_orientations_euler = info['orientations'] 
    joint_positions = info['positions'] 
    joint_names = info['joint_names'] 
    # 'XYZ'
    joint_orientations = R.from_euler('XYZ', joint_orientations_euler, degrees=True).as_quat()
    # 'xyz'
    # joint_orientations = R.from_euler('xyz', joint_orientations_euler, degrees=True).as_quat()
    # 'ZYX'
    # joint_orientations = R.from_euler('ZYX', joint_orientations_euler, degrees=True).as_quat()
    # 'zyx'
    # joint_orientations = R.from_euler('zyx', joint_orientations_euler, degrees=True).as_quat()
    
    viewer.show_pose(joint_names, joint_positions, joint_orientations)
    viewer.run()

def part2():
    """
    part2 我们这部分主要是打印经过多次重复的Scipy的四元数运算，使用Scipy向量化操作和for循环操作的时间
    可以感受一下两者时间的差距
    """
    import time
    info = np.load('one_piece_pose.npy', allow_pickle=True).item()
    joint_orientations_euler = info['orientations'] 
    joint_orientations = R.from_euler('XYZ', joint_orientations_euler, degrees=True)
    test_quat = R.from_rotvec([4, 7, 3])
    joint_num = joint_orientations_euler.shape[0]
    joint_orientations1 = np.zeros((joint_num, 4), dtype=np.float64)
    joint_orientations2 = np.zeros((joint_num, 4), dtype=np.float64)
    nums = 1000
    t1 = time.time()
    for i in range(nums):
        joint_orientations1 = (joint_orientations * test_quat).as_quat()
    t2 = time.time()
    print("vectorized need time {} s.".format(t2 - t1))

    t3 = time.time()
    for i in range(nums):
        for j in range(joint_num):
            joint_orientations2[j] = (joint_orientations[j] * test_quat).as_quat()
    t4 = time.time()
    print("for loop need time {} s.".format(t4 - t3))
    # print("the diff:", np.max(np.abs(joint_orientations1 - joint_orientations2)))

def main():
    viewer = SimpleViewer()

    # 请取消注释需要运行的代码
    # part0
    part0(viewer)
    
    # part1
    # part1(viewer)
    
    # part2
    # part2()


if __name__ == "__main__":
    main()

