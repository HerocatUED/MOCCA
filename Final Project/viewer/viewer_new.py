import os
from direct.showbase.ShowBase import ShowBase
import numpy as np
from panda3d.core import ClockObject
import panda3d.core as pc
import math
from scipy.spatial.transform import Rotation
from direct.showbase.DirectObject import DirectObject
from direct.gui.DirectGui import *
import MoCCASimuBackend
from bvh_loader import BVHMotion
from scipy.spatial.transform import Rotation as R
# from memory_profiler import profile
class CameraCtrl(DirectObject):
    def __init__(self, base, camera):
        super(CameraCtrl).__init__()
        self.accept('mouse1',self.onMouse1Down)
        self.accept('mouse1-up',self.onMouse1Up)
        self.accept('mouse2',self.onMouse2Down)
        self.accept('mouse2-up',self.onMouse2Up)
        self.accept('mouse3',self.onMouse3Down)
        self.accept('mouse3-up',self.onMouse3Up)
        self.accept('wheel_down',self.onMouseWheelDown)
        self.accept('wheel_up',self.onMouseWheelUp)

        self.accept('control-mouse1',self.onMouse1Down)
        self.accept('control-mouse1-up',self.onMouse1Up)
        self.accept('control-mouse2',self.onMouse2Down)
        self.accept('control-mouse2-up',self.onMouse2Up)
        self.accept('control-mouse3',self.onMouse3Down)
        self.accept('control-mouse3-up',self.onMouse3Up)
        self.accept('control-wheel_down',self.onMouseWheelDown)
        self.accept('control-wheel_up',self.onMouseWheelUp)

        self.position = pc.LVector3(4,4,4)
        self.center = pc.LVector3(0,1,0)
        self.up = pc.LVector3(0,1,0)

        self.base = base
        base.taskMgr.add(self.onUpdate, 'updateCamera', sort = 2)
        self.camera = camera
        
        self._locked_info = None
        self._locked_mouse_pos = None
        self._mouse_id = -1
        self.gamepad_x = 0
        self.gamepad_y = 0
        self.has_pad = False
        self.look()
        self._locked_info = (pc.LVector3(self.position), pc.LVector3(self.center), pc.LVector3(self.up))
        
        
    def look(self):    
        self.camera.setPos(self.position)
        self.camera.lookAt(self.center, self.up)

    @property
    def _mousePos(self):
        return pc.LVector2(self.base.mouseWatcherNode.getMouseX(), self.base.mouseWatcherNode.getMouseY())

    def _lockMouseInfo(self):
        self._locked_info = (pc.LVector3(self.position), pc.LVector3(self.center), pc.LVector3(self.up))
        self._locked_mouse_pos = self._mousePos

    def onMouse1Down(self):
        self._lockMouseInfo()
        self._mouse_id = 1

    def onMouse1Up(self):
        self._mouse_id = -1

    def onMouse2Down(self):
        self._lockMouseInfo()
        self._mouse_id = 2

    def onMouse2Up(self):
        self._mouse_id = -1

    def onMouse3Down(self):
        self._lockMouseInfo()
        self._mouse_id = 3

    def onMouse3Up(self):
        self._mouse_id = -1

    def onMouseWheelDown(self):
        z =  self.position - self.center 
        
        scale = 1.1

        if scale < 0.05:
            scale = 0.05

        self.position = self.center + z * scale
        self.look()

    def onMouseWheelUp(self):
        z =  self.position - self.center 
        
        scale = 0.9

        if scale < 0.05:
            scale = 0.05

        self.position = self.center + z * scale
        self.look()
    
    def updateGamepad(self, x, y, task):
        self.gamepad_x = x
        self.gamepad_y = y
        
        self.has_pad = self.gamepad_x**2+self.gamepad_y**2 > 0.04
        
            
    def onUpdate(self, task):
        if self._mouse_id < 0 and not self.has_pad:
            return task.cont
        
        if self.has_pad:
            mousePosOff = pc.LVector2(self.gamepad_x, self.gamepad_y) * 0.02
        else:
            mousePosOff0 = self._mousePos - self._locked_mouse_pos
            mousePosOff = self._mousePos - self._locked_mouse_pos
        
        if self._mouse_id == 1 or self.has_pad:
            if self.has_pad:
                z = self.position - self.center
            else:
                z = self._locked_info[0] - self._locked_info[1]

            zDotUp = self._locked_info[2].dot(z)
            zMap = z - self._locked_info[2] * zDotUp
            angX = math.acos(zMap.length() / z.length()) / math.pi * 180.0

            if zDotUp < 0:
                angX = -angX

            angleScale = 200.0

            x = self._locked_info[2].cross(z)
            x.normalize()
            y = z.cross(x)
            y.normalize()

            rot_x_angle = -mousePosOff.getY() * angleScale
            rot_x_angle += angX
            if rot_x_angle > 85:
                rot_x_angle = 85
            if rot_x_angle < -85:
                rot_x_angle = -85
            rot_x_angle -= angX

            rot_y = pc.LMatrix3()
            rot_y.setRotateMat(-mousePosOff.getX() * angleScale, y, pc.CS_yup_right)
            
            rot_x = pc.LMatrix3()
            rot_x.setRotateMat(-rot_x_angle, x, pc.CS_yup_right)
            if not self.has_pad:
                self.position = self._locked_info[1] + (rot_x * rot_y).xform(z)
            else:
                self.position = self.center + (rot_x * rot_y).xform(z)

        elif self._mouse_id == 2:
            z = self._locked_info[0] - self._locked_info[1]

            shiftScale = 0.5 * z.length()

            x = self._locked_info[2].cross(z)
            z.normalize()
            x.normalize()
            y = z.cross(x)

            shift = x * -mousePosOff.getX() + y* -mousePosOff.getY()
            shift *= shiftScale
            self.position = self._locked_info[0] + shift
            self.center = self._locked_info[1] + shift

        elif self._mouse_id == 3:
            z = self._locked_info[0] - self._locked_info[1]
            
            scale = 1
            scale = 1.0 + scale * mousePosOff0.getY()

            if scale < 0.05:
                scale = 0.05

            self.position = self._locked_info[1] + z * scale

        self.look()

        return task.cont

class SimpleViewer(ShowBase):
    def __init__(self, float_base=False, substep = 32, simu_flag=1, fStartDirect=True, windowType=None):
        '''
        this is only used for my project... lots of assumptions...
        '''
        super().__init__(fStartDirect, windowType)
        self.disableMouse()
        
        self.float_base = float_base
        self.substep = substep
        
        self.camera.lookAt(0,0.9,0)
        self.setupCameraLight()
        self.camera.setHpr(0,0,0)
        
        self.setFrameRateMeter(True)
        globalClock.setMode(ClockObject.MLimited)
        globalClock.setFrameRate(60)
        
        self.load_ground()
        
        xSize = self.pipe.getDisplayWidth()
        ySize = self.pipe.getDisplayHeight()
        props = pc.WindowProperties()
        props.setSize(min(xSize-200, 800), min(ySize-200, 600))
        self.win.requestProperties(props)
        
        # color for links
        color = [131/255,175/255,155/255,1]
        self.tex = self.create_texture(color, 'link_tex')
        
        sceneloader = MoCCASimuBackend.ODESim.JsonSceneLoader()
        simu_type = MoCCASimuBackend.ODESim.ODEScene.SimulationType(0)
        self.scene = sceneloader.load_from_file('stdhuman-scene.json')
        self.scene.set_sim_fps(60 * self.substep)
        self.scene.set_simulation_type(simu_type)
        self.character = self.scene.character0
        self.bvh_name_idx = None
        self.character_to_bvh = None
        self.character_jcnt = len(self.character.joint_info)
        self.character_raw_anchor = self.character.get_raw_anchor()[0].reshape((-1, 3))
        # self.pd_controller = DampedPDControler(self.character)
        self.parent_index = self.character.body_info.parent
        self.load_character()
        self.update_func = None
        self.add_task(self.update, 'update')
        # self.simu_flag = 1 simulation
        # self.simu_flag = 0 not simulation
        self.update_flag = True
        self.simu_flag = simu_flag
        self.accept('space', self.receive_space)
        # self.taskMgr.doMethodLater(1.0, self.simulationTask, "Physics Simulation")
        self.pre_simulation_func = None
        self.add_noise_force = False
        self.additional_force = np.zeros(3)
        self.__additional_force_cnt = 0
    
    def receive_space(self):
        self.update_flag = not self.update_flag
    
    def set_bvh2character(self, motion: BVHMotion):
        self.bvh_name_idx = dict(zip(motion.joint_name, range(len(motion.joint_name))))
        self.character_to_bvh = np.array([self.bvh_name_idx[joint.name] for joint in self.character.joints])
    
    def create_texture(self, color, name):
        img = pc.PNMImage(32,32)
        img.fill(*color[:3])
        img.alphaFill(color[3])
        tex = pc.Texture(name)
        tex.load(img)
        return tex
        
    def load_ground(self):
        self.ground = self.loader.loadModel("material/GroundScene.egg")
        self.ground.reparentTo(self.render)
        self.ground.setScale(100, 1, 100)
        self.ground.setTexScale(pc.TextureStage.getDefault(), 50, 50)
        self.ground.setPos(0, -1, 0)
        
    def setupCameraLight(self):
        # create a orbiting camera
        self.cameractrl = CameraCtrl(self, self.cam)
        self.cameraRefNode = self.camera # pc.NodePath('camera holder')
        self.cameraRefNode.setPos(0,0,0)
        self.cameraRefNode.setHpr(0,0,0)
        self.cameraRefNode.reparentTo(self.render)
        
        self.accept("v", self.bufferViewer.toggleEnable)

        self.d_lights = []
        # Create Ambient Light
        ambientLight = pc.AmbientLight('ambientLight')
        ambientLight.setColor((0.4, 0.4, 0.4, 1))
        ambientLightNP = self.render.attachNewNode(ambientLight)
        self.render.setLight(ambientLightNP)
        
        # Directional light 01
        directionalLight = pc.DirectionalLight('directionalLight1')
        directionalLight.setColor((0.4, 0.4, 0.4, 1))
        directionalLightNP = self.render.attachNewNode(directionalLight)
        directionalLightNP.setPos(10, 10, 10)
        directionalLightNP.lookAt((0, 0, 0), (0, 1, 0))

        directionalLightNP.wrtReparentTo(self.cameraRefNode)
        self.render.setLight(directionalLightNP)
        self.d_lights.append(directionalLightNP)
        
        # Directional light 02
        directionalLight = pc.DirectionalLight('directionalLight2')
        directionalLight.setColor((0.4, 0.4, 0.4, 1))
        directionalLightNP = self.render.attachNewNode(directionalLight)
        directionalLightNP.setPos(-10, 10, 10)
        directionalLightNP.lookAt((0, 0, 0), (0, 1, 0))

        directionalLightNP.wrtReparentTo(self.cameraRefNode)
        self.render.setLight(directionalLightNP)
        self.d_lights.append(directionalLightNP)
        
        
        # Directional light 03
        directionalLight = pc.DirectionalLight('directionalLight3')
        directionalLight.setColorTemperature(6500)        
        directionalLightNP = self.render.attachNewNode(directionalLight)
        directionalLightNP.setPos(0, 20, -10)
        directionalLightNP.lookAt((0, 0, 0), (0, 1, 0))

        directionalLightNP.wrtReparentTo(self.cameraRefNode)
        directionalLight.setShadowCaster(True, 2048, 2048)
        directionalLight.getLens().setFilmSize((10,10))
        directionalLight.getLens().setNearFar(0.1,300)
        
        self.render.setLight(directionalLightNP)
        self.d_lights.append(directionalLightNP)

        self.render.setShaderAuto(True)
    
    def create_joint(self, link_id, position, end_effector=False):
        # create a joint
        box = self.loader.loadModel("material/GroundScene.egg")
        node = self.render.attachNewNode(f"joint{link_id}")
        box.reparentTo(node)
        
        # add texture
        box.setTextureOff(1)
        if end_effector:
            tex = self.create_texture([0,1,0,1], f"joint{link_id}_tex")
            box.setTexture(tex, 1)
        box.setScale(0.01,0.01,0.01)
        node.setPos(self.render, *position)
        return node
    
    def create_link(self, link_id, position, scale, rot):
        # create a link
        box = self.loader.loadModel("material/GroundScene.egg")
        node = self.render.attachNewNode(f"link{link_id}")
        box.reparentTo(node)
        
        # add texture
        box.setTextureOff(1)
        box.setTexture(self.tex,1)
        box.setScale(*scale)
        
        node.setPos(self.render, *position)
        if rot is not None:
            node.setQuat(self.render, pc.Quat(*rot[[3,0,1,2]].tolist()))
        return node
    
    def show_axis_frame(self):
        pose = [ [1,0,0], [0,1,0], [0,0,1] ]
        color = [ [1,0,0,1], [0,1,0,1], [0,0,1,1] ]
        for i in range(3):
            box = self.loader.loadModel("material/GroundScene.egg")
            box.setScale(0.1, 0.1, 0.1)
            box.setPos(*pose[i])
            tex = self.create_texture(color[i], f"frame{i}")
            box.setTextureOff(1)
            box.setTexture(tex,1)
            box.reparentTo(self.render)
    
    def update(self, task):
        if self.update_func and self.update_flag:
            self.update_func(self)
        if self.update_flag:
            if self.simu_flag:
                self.simulationTask(rendering=True)
            else:
                self.kinematicsTask(rendering=True)
        return task.cont
    
    def kinematicsTask(self, rendering=True):
        if rendering:
            self.sync_physics_to_kinematics()
        pass

    def simulationTask(self, pre_simulation_func = None, rendering = False):
        if pre_simulation_func is None:
            pre_simulation_func = self.pre_simulation_func
        for i in range(self.substep):
            if pre_simulation_func:
                pre_simulation_func()
            self.scene.simu_func()
        if rendering:
            self.sync_physics_to_kinematics()
            
        return 
    
    def sync_physics_to_kinematics(self): # not done
        joint_positions = self.get_physics_joint_positions()
        body_positions = self.get_physics_body_positions()
        body_orientations = self.get_physics_body_orientations()
        for i in range(len(self.character.bodies)):
            self.joint[i].setPos(self.render, *joint_positions[i].tolist())
            self.body[i].setPos(self.render, *body_positions[i].tolist())
            self.body[i].setQuat(self.render, pc.Quat(*body_orientations[i,[3,0,1,2]].tolist()))
        
    def load_character(self):
        # info = np.load('character_model.npy', allow_pickle=True).item()
        # joint_pos = info['joint_pos']
        # body_pos = info['body_pos']
        # joint_name =  info['joint_name']
        # body_rot = info.get('body_ori', None)
        
        joint, body = [], []
        
        thickness = 0.03
        joint_name = ['RootJoint'] + self.character.get_joint_names()
        name_idx_map = {joint_name[i]: i for i in range(len(joint_name))}
        scale = [[thickness]*3 for i in range(len(self.character.bodies))]
        
        scale[name_idx_map['RootJoint']] = [0.06,0.06,0.03]
        scale[name_idx_map['torso_head']] = [0.05,0.08,thickness]
        scale[name_idx_map['lowerback_torso']] = [thickness,0.04,thickness]
        scale[name_idx_map['pelvis_lowerback']] = [thickness,0.035,thickness]
        
        scale[name_idx_map['lHip']] = scale[name_idx_map['rHip']] = [thickness,0.15,thickness]
        scale[name_idx_map['rKnee']] = scale[name_idx_map['lKnee']] = [thickness,0.16,thickness]
        scale[name_idx_map['rAnkle']] = scale[name_idx_map['lAnkle']] = [thickness*1.5,thickness,0.08]

        scale[name_idx_map['rTorso_Clavicle']] = scale[name_idx_map['lTorso_Clavicle']] = [0.05,thickness,thickness]
        scale[name_idx_map['rShoulder']] = scale[name_idx_map['lShoulder']] = [0.07,thickness,thickness]
        scale[name_idx_map['rElbow']] = scale[name_idx_map['lElbow']] = [0.07,thickness,thickness]
        
        
        # physics_body = []
        # physics_joint = []
        # damping_joint = []
        # self.world = OdeWorld()
        # self.world.setGravity(0, -9.81, 0)
        # #self.world.set_cfm(0.00001)
        # self.world.initSurfaceTable(2)
        # self.world.setSurfaceEntry(0, 0, 1.5, 0.0, 0, 0.9, 0.001, 0.0, 0.000)
        # self.space = OdeSimpleSpace()
        # self.contactgroup = OdeJointGroup()
        # self.space.setAutoCollideWorld(self.world)
        # self.space.setAutoCollideJointGroup(self.contactgroup)
        # groundGeom = OdePlaneGeom(self.space, pc.Vec4(0, 1, 0, 0))
        # groundGeom.setCollideBits(pc.BitMask32(0x00000001))
        # groundGeom.setCategoryBits(pc.BitMask32(0x00000001))

        # joint_pos = np.concatenate([body_pos[0:1], joint_pos], axis=0)
        joint_pos = self.get_physics_joint_positions()
        body_pos = self.get_physics_body_positions()
        body_rot = self.get_physics_body_orientations()
        for i in range(len(self.character.bodies)):
            joint.append(self.create_joint(i, joint_pos[i], 'end' in joint_name[i]))
            # if i < body_pos.shape[0]:
            body.append(self.create_link(i, body_pos[i], scale[i], rot = body_rot[i]))
                # body[-1].wrtReparentTo(joint[-1])
                # physics-body
        #         ode_body = OdeBody(self.world)
        #         ode_body.setPosition(body[-1].getPos(self.render))
        #         ode_body.setQuaternion(body[-1].getQuat(self.render))
        #         mass = OdeMass()
        #         if  'Toe' in joint_name[i]:
        #             mass.setBox(100, *[ j * 10 for j in scale[i]])
        #         elif  'Torso_Clavicle' in joint_name[i]:
        #             mass.setBox(50, * [ j * 10 for j in scale[i]])
        #         else:
        #             mass.setBox(50, * [ j * 10 for j in scale[i]])
        #         ode_body.setMass(mass)
        #         total_mass += mass.getMagnitude()

        #         # geometry for collision
        #         boxGeom = OdeBoxGeom(self.space, *[x * 1.5 for x in scale[i]])
        #         boxGeom.set_position(body[-1].getPos(self.render))
        #         boxGeom.setBody(ode_body)
        #         if not 'Toe' in joint_name[i]:
        #             boxGeom.setCollideBits(pc.BitMask32(0x00000001))
        #             boxGeom.setCategoryBits(pc.BitMask32(0x00000001))
                    
        #         # physics joints
        #         if 'Toe' in joint_name[i] or 'Elbow' in joint_name[i] or 'Knee' in joint_name[i]:
        #             odejoint = OdeHingeJoint(self.world)
        #             if 'Elbow' in joint_name[i]:
        #                 odejoint.setAxis((0, 1, 0))
        #             else:
        #                 odejoint.setAxis((1, 0, 0))
        #         else:
        #             odejoint = OdeBallJoint(self.world)
        #         if i !=0:
        #             odejoint.attach(ode_body, physics_body[info['parent'][i]])
        #         else:
        #             if not self.float_base:
        #                 odejoint.attach(ode_body, None)
        #             pass
        #         odejoint.setAnchor(*joint_pos[i])
        #         physics_body.append(ode_body)
        #         physics_joint.append(odejoint)
                
        self.body = body
        # self.physics_body = physics_body
        # self.damping_joint = damping_joint
        # self.physics_joint = physics_joint
        
        # real_len = body_pos.shape[0]
        self.joint = joint
        self.joint_name = ['RootJoint'] + self.character.joint_info.joint_names()
        self.name2idx = name_idx_map
        # self.parent_index = info['parent'][:real_len]
        # self.init_joint_pos = self.get_joint_positio ns()

    def joint2body(self, joint_pos_, joint_quat_): # maybe bug here
        # joint_pos = joint_pos_[self.character_to_bvh, :]
        joint_pos = joint_pos_[1:]
        # joint_quat = joint_quat_[self.character_to_bvh, :]
        joint_quat = joint_quat_[1:]
        child_body_pos = np.zeros((self.character_jcnt, 3))
        for jidx in range(self.character_jcnt):
            rot = Rotation(joint_quat[jidx, :], copy=False)
            child_body_pos[jidx, :] = joint_pos[jidx, :] - rot.apply(self.character_raw_anchor[jidx])
        body_pos = np.zeros((self.character_jcnt + 1, 3))
        body_quat = np.zeros((self.character_jcnt + 1, 4))
        body_pos[self.character.body_info.root_body_id, :] = joint_pos_[0]
        body_pos[self.character.joint_to_child_body, :] = child_body_pos
        body_quat[self.character.body_info.root_body_id, :] = joint_quat_[0]
        body_quat[self.character.joint_to_child_body, :] = joint_quat
        return body_pos, body_quat
    
    def set_pose(self, joint_name, pos_, quat_, with_noise=False):
        if with_noise:
            pose_noise = np.random.randn(len(self.body), 3) * 0.1
            noise = Rotation.from_rotvec(pose_noise)
            quat_ = (noise * Rotation.from_quat(quat_)).as_quat()
        c_id = self.character.body_info.body_c_id
        pos, quat = self.joint2body(pos_, quat_)
        self.character.world.loadBodyPos(c_id, pos.flatten())
        self.character.world.loadBodyQuat(c_id, quat.flatten())
        pass

    def get_pose(self): # done
        return self.character.get_body_quat(), self.character.get_body_ang_velo()
    
    # def sync_kinematic_to_physics(self):
    #     for i in range(len(self.physics_body)):
    #         self.physics_body[i].setPosition(self.body[i].getPos(self.render))
    #         self.physics_body[i].setQuaternion(self.body[i].getQuat(self.render))        
    
    def set_body_positions(self, body_pos): # new
        self.character.body_info.set_body_pos(body_pos)
    
    def set_body_orientations(self, body_quat): # new
        self.character.body_info.set_body_quat(body_quat)

    def set_body_velocities(self, body_vel): # done
        self.character.body_info.set_body_velo(body_vel)
        
    def set_body_angular_velocities(self, body_avel): # done
        self.character.body_info.set_body_ang_velo(body_avel)
    
    @property
    def root_pos(self): # done
        return self.character.root_body_pos
    
    @property
    def root_quat(self): # done
        return self.character.root_body_quat
    
    def get_root_pos_vel(self): # done
        return self.root_pos, self.character.root_body.LinearVelNumpy
    
    def get_physics_joint_positions(self): # done
        pos = self.character.joint_info.get_global_anchor1()
        return np.concatenate([self.root_pos[None], pos], axis=0)
    
    def get_physics_body_positions(self): # done
        return self.character.get_body_pos()
    
    def get_physics_body_orientations(self): # new
        return self.character.get_body_quat()

    def get_body_mass(self): # done
        return self.character.body_info.mass_val
    
    def get_body_velocities(self): # done
        return self.character.get_body_velo()
    
    def get_body_angular_velocities(self): # done
        return self.character.get_body_ang_velo()
    
    def get_physics_joint_orientations(self): # done
        _, joint_qs, _, _ = self.character.joint_info.get_parent_child_qs()
        return joint_qs
    
    # def get_joint_avel_by_name(self, name):
    #     return np.array(self.physics_body[self.name2idx[name]].getAngularVel())
    
    # def get_physics_joint_position_by_name(self, name):
    #     if name == 'RootJoint':
    #         pos = self.physics_body[0].getPosition()
    #     else:
    #         pos = self.physics_joint[self.name2idx[name]].getAnchor()
    #     return np.array(pos)
    
    # def get_physics_joint_orientation_by_name(self, name):
    #     quat = self.physics_body[self.name2idx[name]].getQuaternion()
    #     return np.array(quat)[..., [1,2,3,0]]
    
    # def set_joint_position_by_name(self, name, pos):
    #     self.joints[self.name2idx[name]].setPos(self.render, *pos)
    
    # def set_joint_orientation_by_name(self, name, quat):
    #     self.joints[self.name2idx[name]].setQuat(self.render, pc.Quat(*quat[...,[3,0,1,2]].tolist()))
    
    # def set_joint_position_orientation(self, link_name, pos, quat): 
    #     if not link_name in self.name2idx or self.name2idx[link_name] >= len(self.joint):
    #         return
    #     self.set_joints_with_idx(self.name2idx[link_name], pos, quat)
    
    # def set_joints_with_idx(self, idx, pos, quat):
    #     self.joint[idx].setPos(self.render, *pos.tolist())
    #     self.joint[idx].setQuat(self.render, pc.Quat(*quat[...,[3,0,1,2]].tolist()))
    
    def set_torque(self, torque): # done
        # assert torque.shape[0] == self.character_jcnt
        torque = np.ascontiguousarray(torque, dtype=np.float64)
        
        for i in range(1, len(self.character.bodies)):
            # start from, ignore root torques completely
            self.character.bodies[i].addTorqueNumpy(torque[i])
            self.character.bodies[self.parent_index[i]].addTorqueNumpy(-torque[i])
    
    def set_root_force(self, force): # done
        force_all = np.zeros((len(self.character.bodies), 3))
        force_all[0] = force
        force = np.ascontiguousarray(force, dtype=np.float64)
        self.character.body_info.add_body_force(force_all)
                
    def set_root_torque(self, torque): # done
        self.character.bodies[0].addTorqueNumpy(torque)
    
    def add_horizontal_force(self):
        if not self.add_noise_force:
            return
        
        if self.__additional_force_cnt % self.scene.sim_fps * 1 == 0:
            angle = np.random.uniform(0, np.pi*2)
            mag = 10
            self.additional_force = np.array((mag * np.sin(angle), 0, mag*np.cos(angle)))
                        
        self.__additional_force_cnt += 1
        
        if self.root_pos[1] > 0.2:
            self.set_root_force(self.additional_force)
                
        
    # def show_pose(self, joint_name_list, joint_positions, joint_orientations):
    #     length = len(joint_name_list)
    #     assert joint_positions.shape == (length, 3)
    #     assert joint_orientations.shape == (length, 4)
        
    #     for i in range(length):
    #         self.set_joint_position_orientation(joint_name_list[i], joint_positions[i], joint_orientations[i])
    
    # def show_rest_pose(self, joint_name, joint_parent, joint_offset):
    #     length = len(joint_name)
    #     joint_positions = np.zeros((length, 3), dtype=np.float64)
    #     joint_orientations = np.zeros((length, 4), dtype=np.float64)
    #     for i in range(length):
    #         if joint_parent[i] == -1:
    #             joint_positions[i] = joint_offset[i]
    #         else:
    #             joint_positions[i] = joint_positions[joint_parent[i]] + joint_offset[i]
    #         joint_orientations[i, 3] = 1.0
    #         self.set_joint_position_orientation(joint_name[i], joint_positions[i], joint_orientations[i])

    # def get_meta_data(self):
        # return self.joint_name, self.parent_index, self.init_joint_pos
    
    def move_marker(self, marker, x, y):
        
        if not self.update_marker_func:
            return
        
        y_axis = self.cameractrl._locked_info[2]
        z_axis = self.cameractrl.position - self.cameractrl.center
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        pos = np.array(marker.getPos(self.render))
        pos += x_axis * x + y_axis * y
        marker.setPos(self.render, *pos.tolist())
        self.update_marker_func(self)
    
    def camera_fwd(self):
        return self.cameractrl.center - self.cameractrl.position
    
    def create_marker(self, pos, color):
        self.update_marker_func = None
        marker = self.loader.loadModel("material/GroundScene.egg")
        marker.setScale(0.05,0.05,0.05)
        marker.setPos(*pos)
        tex = self.create_texture(color, "marker")
        marker.setTextureOff(1)
        marker.setTexture(tex,1)
        
        marker.wrtReparentTo(self.render)
        
        self.accept('w', self.move_marker, [marker, 0, 0.05])
        self.accept('s', self.move_marker, [marker, 0, -0.05])
        self.accept('a', self.move_marker, [marker, -0.05, 0])
        self.accept('d', self.move_marker, [marker, 0.05, 0])
        
        self.accept('w-repeat', self.move_marker, [marker, 0, 0.05])
        self.accept('s-repeat', self.move_marker, [marker, 0, -0.05])
        self.accept('a-repeat', self.move_marker, [marker, -0.05, 0])
        self.accept('d-repeat', self.move_marker, [marker, 0.05, 0])
        return marker
    
    def create_marker2(self, pos, color):
        self.update_marker_func = None
        marker = self.loader.loadModel("material/GroundScene.egg")
        marker.setScale(0.05,0.05,0.05)
        marker.setPos(*pos)
        tex = self.create_texture(color, "marker")
        marker.setTextureOff(1)
        marker.setTexture(tex,1)
        
        marker.wrtReparentTo(self.render)
        
        self.accept('arrow_up', self.move_marker, [marker, 0, 0.05])
        self.accept('arrow_down', self.move_marker, [marker, 0, -0.05])
        self.accept('arrow_left', self.move_marker, [marker, -0.05, 0])
        self.accept('arrow_right', self.move_marker, [marker, 0.05, 0])
        
        self.accept('arrow_up-repeat', self.move_marker, [marker, 0, 0.05])
        self.accept('arrow_down-repeat', self.move_marker, [marker, 0, -0.05])
        self.accept('arrow_left-repeat', self.move_marker, [marker, -0.05, 0])
        self.accept('arrow_right-repeat', self.move_marker, [marker, 0.05, 0])
        return marker
    
    def create_arrow(self, pos, forward_xz = np.array([0,1]), color = [1,0,0,0]):
        from .visualize_utils import draw_arrow
        arrow = self.render.attachNewNode("arrow")
        draw_arrow(arrow, 0.3, 1, color)
        arrow.setPos(*pos)
        
        from scipy.spatial.transform import Rotation as R
        axis = np.array([0,1,0])
        angle = np.arctan2(forward_xz[0], forward_xz[1])
        rot = R.from_rotvec(angle * axis).as_quat()
        quat = pc.Quat(rot[3], rot[0], rot[1], rot[2])
        arrow.setQuat(quat)
        return arrow
    