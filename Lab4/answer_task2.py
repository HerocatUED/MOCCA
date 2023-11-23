from bvh_utils import *
#---------------你的代码------------------#
# translation 和 orientation 都是全局的
def dq_skinning(joint_translation, joint_orientation, T_pose_joint_translation, T_pose_vertex_translation, skinning_idx, skinning_weight):
    """
    skinning函数，给出一桢骨骼的位姿，计算蒙皮顶点的位置
    假设M个关节，N个蒙皮顶点，每个顶点受到最多4个关节影响
    输入：
        joint_translation: (M,3)的ndarray, 目标关节的位置
        joint_orientation: (M,4)的ndarray, 目标关节的旋转，用四元数表示
        T_pose_joint_translation: (M,3)的ndarray, T pose下关节的位置
        T_pose_vertex_translation: (N,3)的ndarray, T pose下蒙皮顶点的位置
        skinning_idx: (N,4)的ndarray, 每个顶点受到哪些关节的影响（假设最多受4个关节影响）
        skinning_weight: (N,4)的ndarray, 每个顶点受到对应关节影响的权重
    输出：
        vertex_translation: (N,3)的ndarray, 蒙皮顶点的位置
    """
    vertex_translation = np.zeros_like(T_pose_vertex_translation)

    # 默认使用 task1
    from answer_task1 import skinning
    vertex_translation = skinning(joint_translation, joint_orientation, T_pose_joint_translation, T_pose_vertex_translation, skinning_idx, skinning_weight)
    
    #---------------------------你的代码---------------------------#
    N = np.shape(T_pose_vertex_translation)[0]
    for i in range(N):
        dq = np.zeros((2,4))
        for j in range(4):
            idx = skinning_idx[i, j]
            trans = joint_translation[idx] - R.from_quat(joint_orientation[idx]).apply(T_pose_joint_translation[idx])
            dq += skinning_weight[i,j] * QuatTrans2UDQ(joint_orientation[idx], trans)
        norm = np.dot(dq[0],dq[0])**0.5
        dq /= norm
        m = cal_M(dq)
        v = np.ones((4,1))
        v[[0,1,2],0] = T_pose_vertex_translation[i][[0,1,2]]
        vertex_translation[i] = (m @ v).reshape((3,))
    #-------------------------------------------------------------#
    return vertex_translation


def QuatTrans2UDQ(q0, t):
    q = np.copy(q0)
    q[0],q[1],q[2],q[3] = q[3],q[0],q[1],q[2]
    dq = np.zeros((2,4))
    dq[0] = np.copy(q)
    dq[1][0] = -0.5*(t[0]*q[1] + t[1]*q[2] + t[2]*q[3])
    dq[1][1] = 0.5*( t[0]*q[0] + t[1]*q[3] - t[2]*q[2])
    dq[1][2] = 0.5*(-t[0]*q[3] + t[1]*q[0] + t[2]*q[1])
    dq[1][3] = 0.5*( t[0]*q[2] - t[1]*q[1] + t[2]*q[0])
    return dq

def cal_M(dq):
    m = np.zeros((3,4))
    w,x,y,z = dq[0,0], dq[0,1], dq[0,2], dq[0,3]
    m[0,0] = 1- 2*y**2 - 2*z**2
    m[0,1] = 2*x*y - 2*w*z
    m[0,2] = 2*x*z + 2*w*y
    m[0,3] = 2.0*(-dq[1][0]*dq[0][1] + dq[1][1]*dq[0][0] - dq[1][2]*dq[0][3] + dq[1][3]*dq[0][2])
    m[1,0] = 2*x*y + 2*w*z
    m[1,1] = 1 - 2*x**2 - 2*z**2
    m[1,2] = 2*y*z - 2*w*x
    m[1,3] = 2.0*(-dq[1][0]*dq[0][2] + dq[1][1]*dq[0][3] + dq[1][2]*dq[0][0] - dq[1][3]*dq[0][1])
    m[2,0] = 2*x*z - 2*w*y
    m[2,1] = 2*y*z + 2*w*x
    m[2,2] = 1 - 2*x**2 - 2*y**2
    m[2,3] = 2.0*(-dq[1][0]*dq[0][3] - dq[1][1]*dq[0][2] + dq[1][2]*dq[0][1] + dq[1][3]*dq[0][0])
    return m


