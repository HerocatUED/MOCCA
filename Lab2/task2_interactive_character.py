from viewer.controller import SimpleViewer, Controller
from answer_task2 import CharacterController
import numpy as np

class InteractiveUpdate():
    def __init__(self, viewer, controller, character_controller):
        self.viewer = viewer
        self.controller = controller
        self.character_controller = character_controller
        
    def update(self, task):
        desired_pos_list, desired_rot_list, desired_vel_list, desired_avel_list, current_gait = \
            self.controller.get_desired_state()
        
        # 逐帧更新角色状态
        character_state = self.character_controller.update_state(
                desired_pos_list, desired_rot_list, 
                desired_vel_list, desired_avel_list
                )
        # 同步手柄和角色的状态
        self.sync_controller_and_character(character_state)
        for i in range(len(character_state[0])):
            name, pos, rot = character_state[0][i], character_state[1][i], character_state[2][i]
            self.viewer.set_joint_position_orientation(name, pos, rot)
        return task.cont   

    def sync_controller_and_character(self, character_state):
        '''
        这一部分用于同步手柄和你的角色的状态
        更新后很有可能会出现手柄和角色的位置不一致，
        这里用一个简单的方案，将手柄的位置对齐于角色我位置
        '''
        controller_pos = character_state[1][0] 
        self.controller.set_pos(controller_pos)

def main():
    viewer = SimpleViewer()
    controller = Controller(viewer)
    character_controller = CharacterController(controller)
    task = InteractiveUpdate(viewer, controller, character_controller)
    viewer.addTask(task.update)
    viewer.run()
    pass

if __name__ == '__main__':
    main()