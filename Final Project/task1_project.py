from viewer.controller import  Controller
from viewer.viewer_new import SimpleViewer
from answer_project import *
import numpy as np

class InteractiveUpdate():
    def __init__(self, viewer, controller, character_controller):
        self.viewer = viewer
        self.controller = controller
        self.character_controller = character_controller
        self.reset()
        # 对于physics-based方法不能set角色状态，而是应该调整施加给character的力
        # 你可以设置成你需要的任意函数，此函数会在physics step时候调用，
        # 详见SimpleViewer的simulationTask函数
        self.viewer.pre_simulation_func = self.character_controller.pd_controller.apply_root_force_and_torque

        
    def reset(self):
        # 一个简单的reset函数，把character设置成motion[0]第一帧
        motion = self.character_controller.motions[0]
        motion.adjust_joint_name(self.controller.viewer.joint_name)
        joint_translation, joint_orientation = motion.batch_forward_kinematics(root_pos=self.viewer.root_pos)
        joint_name = motion.joint_name
        joint_translation = joint_translation[0]
        joint_orientation = joint_orientation[0]
        self.viewer.set_pose(joint_name, joint_translation, joint_orientation)
        self.viewer.sync_physics_to_kinematics()
        
        
    def update(self, task):
        desired_pos_list, desired_rot_list, desired_vel_list, desired_avel_list, current_gait = \
            self.controller.get_desired_state()
        
        # 逐帧更新角色状态
        character_state = self.character_controller.update_state(
                desired_pos_list, desired_rot_list, 
                desired_vel_list, desired_avel_list
                )
        # 同步手柄和角色的状态
        self.character_controller.sync_controller_and_character(character_state)
        
        # viewer 渲染
        # 对于kinematics-based方法直接set角色状态
        if not self.viewer.simu_flag:
            name, pos, rot = character_state[0], character_state[1], character_state[2]
            self.viewer.set_pose(name, pos, rot)
        # 对于physics-based方法不能set角色状态，而是应该调整施加给character的力
        # 对应init函数里面的pre_simulation_func选项
        
            
        return task.cont   

    

def main():
    # 创建一个控制器，是输入信号的一层包装与平滑
    # 可以使用键盘(方向键或wasd)或鼠标控制视角
    # 对xbox类手柄的支持在windows10下测试过，左手柄控制移动，右手柄控制视角
    # 其余手柄(如ps手柄)不能保证能够正常工作
    # 注意检测到手柄后，键盘输入将被忽略
    # simu_flag = 0 表示不使用physics character
    # simu_flag = 1 表示使用physics character
    viewer = SimpleViewer(float_base = True, substep = 32, simu_flag=0)
    pd_controller = PDController(viewer)
    controller = Controller(viewer)
    character_controller = CharacterController(viewer, controller, pd_controller)
    task = InteractiveUpdate(viewer, controller, character_controller)
    viewer.addTask(task.update)
    viewer.run()
    pass

if __name__ == '__main__':
    main()