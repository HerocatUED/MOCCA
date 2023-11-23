import numpy as np
from typing import List, Dict
from bvh_motion import BVHMotion

class Edge:
    def __init__(self, label:str, dest:'Node'):
        # 边的名字，对应Transition动画的名字
        # 比如说'motion1.bvh->motion2.bvh'
        self.label = label
        # 边的末端节点
        self.destination = dest

class Node:
    def __init__(self, id:int=-1, name:str=None, motion:BVHMotion=None):
        # 节点的编号
        self.identity = id
        # 以本节点连出的边的集合
        self.edges : List[Edge] = []
        # 节点的名字
        # 一般是动画的名字，如'motion1.bvh'
        self.name = name if name is not None else str(self.identity)
        # 动作片段
        self.motion = motion

    @property
    def n_edges(self) -> int:
        return len(self.edges)
    
    # 添加新的边
    def add_edge(self, input_edge:Edge):
        self.edges.append(input_edge)
    
    # 删除对应编号的边
    # 注意这个编号指的是在self._edges的内部编号
    def remove_edge(self, edge_id:int):
        self.edges.pop(edge_id)

    # 得到对应编号的边
    # 注意这个编号指的是在self._edges的内部编号  
    def get_edge(self, edge_id:int):
        return self.edges[edge_id]
    
class Graph:
    def __init__(self, graph_file:str=None) -> None:
        # 节点的集合
        self.nodes : List [Node] = []
        # 节点连接图的file name
        self.graph_file = graph_file
        # 动画集合，以BVHMotion类存储的集合
        self.motions : List[BVHMotion] = []
        # 动画file name 文件夹地址
        self.animation_dir = './motion_material/'

    @property
    def n_nodes(self) -> str:
        return len(self.nodes)
    
    # 图上添加节点
    def add_node(self, node: Node):
        node.identity = self.n_nodes
        self.nodes.append(node)
        return node.identity
    
    # 修改节点
    def change_node(self, idx, node: Node):
        self.nodes[idx] = node

    # 读取节点连接关系，并且以BVHMotion类存储初始动画clip
    def load_from_file(self, graph_file=None):
        # load node and edge structure
        if not graph_file:
            graph_file = self.graph_file
            
        info = np.load(graph_file, allow_pickle=True).item()
        node_id = info['node_id'] 
        node_name = info['node_name'] 
        node_edges_name = info['node_edges_name'] 
        node_edges_dest = info['node_edges_dest'] 
                
        nd_list = [Node(_id, _name) for _id, _name in zip(node_id, node_name)]
                    
        for nd, eg_names, eg_dests in zip(nd_list, node_edges_name, node_edges_dest):
            nd.edges = [Edge(label=_name, dest=nd_list[node_name.index(_dest)])
                        for _name, _dest in zip(eg_names, eg_dests)]
                
        self.nodes = nd_list
        
        # load animation clip 
        for node in self.nodes:
            bvh_file_name = self.animation_dir + node.name
            bvh = BVHMotion(bvh_file_name)
            self.motions.append(bvh)
            node.motion = bvh

    # 绘制图的节点和边的连接关系
    def draw_structure_graph(self):
        """
        如果需要查看Graph节点连接关系，可以使用本函数利用第三方库实现
        [graphviz](https://graphviz.gitlab.io/download/)，
        值得注意的是graphviz安装分为软件安装和python安装(pip install graphviz)，
        并请将其安装目录放入环境PATH变量，
        附一个[graphviz安装帖子](https://zhuanlan.zhihu.com/p/268532582).
        """ 
        from graphviz import Digraph
        dot = Digraph()
        node_set = []
        edge_set = []
        for node in self.nodes:
            ori_node_name = str(node.identity)
            if ori_node_name not in node_set:
                node_set.append(ori_node_name)
                # dot.node(ori_node_name, "Node"+ori_node_name+"--"+node._name)
                dot.node(ori_node_name, node.name)
            for edge in node.edges:
                
                dest_node = edge.destination
                dest_node_name = str(dest_node.identity)
                edge_name = edge.label
                if dest_node_name not in node_set:
                    node_set.append(dest_node_name)
                    # dot.node(dest_node_name, "Node"+dest_node_name+"--"+dest_node._name)
                    dot.node(dest_node_name, dest_node.name)
                if edge_name not in edge_set:
                    edge_set.append(edge_name)
                    dot.edge(ori_node_name, dest_node_name, label=edge_name)


        dot.render(view=True)

if __name__ == "__main__":
    nd_list = []
    for i in range(10):
        nd = Node(id=i,name=str(i))
        nd_list.append(nd)

    for i in range(10):    
        rand_nd = np.random.randint(0, 10)
        nd_list[i].add_edge(Edge(label=str(i),dest=nd_list[rand_nd]))

    graph = Graph()
    graph.nodes = nd_list
    # graph.draw_structure_graph()
