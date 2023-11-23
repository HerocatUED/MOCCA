import numpy as np

# part 0
def load_meta_data(bvh_file_path):
    """
    请把lab1-FK-part1的代码复制过来
    请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        channels: List[int]，整数列表，joint的自由度，根节点为6(三个平动三个转动)，其余节点为3(三个转动)
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量
    Tips:
        joint_name顺序应该和bvh一致
    """
    

    joint_name = None
    joint_parent = None
    channels = None
    joint_offset = None

    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        cnt = 0
        parent = [0]
        for i in range(len(lines)):
            if lines[i].startswith('ROOT'):
                joint_name = [lines[i][lines[i].find("ROOT")+4:].strip()]
                joint_parent = [-1]
                i += 2
                data = [float(x) for x in lines[i][lines[i].find("OFFSET")+6:].split()]
                joint_offset = [np.array(data).reshape(1, -1)]
                i += 1
                channels = [int(lines[i][lines[i].find("CHANNELS")+9])]
                break
        for line in lines[i+1:]:
            if "JOINT" in line:
                joint_name.append(line[line.find("JOINT")+5:].strip())
                joint_parent.append(parent[-1])
                cnt += 1
            elif "OFFSET" in line:
                data = [float(x) for x in line[line.find("OFFSET")+6:].split()]
                joint_offset.append(np.array(data).reshape(1, -1))
            elif "CHANNELS" in line:
                channels.append(int(line[line.find("CHANNELS")+9]))
            elif "{" in line:
                parent.append(cnt)
            elif "}" in line:
                parent.pop()
        joint_offset = np.concatenate(joint_offset, axis=0)  
    
    return joint_name, joint_parent, channels, joint_offset