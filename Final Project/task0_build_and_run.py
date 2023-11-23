from viewer.viewer_new import SimpleViewer
from viewer.controller import Controller

def main():
    viewer = SimpleViewer(float_base = True, substep = 32, simu_flag=0)
    # 创建一个控制器，是输入信号的一层包装与平滑
    # 可以使用键盘(方向键或wasd)或鼠标控制视角
    # 对xbox类手柄的支持在windows10下测试过，左手柄控制移动，右手柄控制视角
    # 其余手柄(如ps手柄)不能保证能够正常工作
    # 注意检测到手柄后，键盘输入将被忽略
    # simu_flag = 0 表示不使用physics character
    # simu_flag = 1 表示使用physics character
    controller = Controller(viewer)
    viewer.run()

if __name__ == '__main__':
    main()