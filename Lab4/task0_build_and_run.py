from Viewer.mesh_viewer import MeshViewer

def main():
    viewer = MeshViewer()
    print(viewer.init_abs_position)
    viewer.run()

if __name__ == '__main__':
    main()