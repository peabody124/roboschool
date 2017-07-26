from roboschool.gym_forward_walker import RoboschoolForwardWalker
from roboschool.gym_urdf_robot_env import RoboschoolUrdfEnv
from roboschool.scene_abstract import cpp_household
from roboschool.scene_stadium import SinglePlayerStadiumScene
import numpy as np

class RoboschoolQuadcopterHover(RoboschoolForwardWalker, RoboschoolUrdfEnv):
    random_yaw = False
    foot_list = ["FL", "FR", "BL", "BR"]

    def __init__(self):
        RoboschoolForwardWalker.__init__(self, power=0.30)
        RoboschoolUrdfEnv.__init__(self,
            "quadcopter_description/urdf/quadcopter-v1.urdf",
            "center",
            action_dim=4, obs_dim=6,
            fixed_base=False,
            self_collision=False)

    def create_single_player_scene(self):
        return SinglePlayerStadiumScene(gravity=9.8, timestep=0.0165/8, frame_skip=8)   # 8 instead of 4 here

    def alive_bonus(self, z, pitch):
        x,y,z = self.base.pose().xyz()
        print("Position: " + self.base.pose().xyz())

        return 1 # TODO: bonus for at certain position

    def robot_specific_reset(self):
        RoboschoolForwardWalker.robot_specific_reset(self)
        self.set_initial_orientation(yaw_center=0, yaw_random_spread=np.pi)
        self.base = self.parts["center"]

    random_yaw = False

    def set_initial_orientation(self, yaw_center, yaw_random_spread):
        cpose = cpp_household.Pose()
        if not self.random_yaw:
            yaw = yaw_center
        else:
            yaw = yaw_center + self.np_random.uniform(low=-yaw_random_spread, high=yaw_random_spread)

        cpose.set_xyz(self.start_pos_x, self.start_pos_y, self.start_pos_z + 1.0)
        cpose.set_rpy(0, 0, yaw)  # just face random direction, but stay straight otherwise
        self.cpp_robot.set_pose_and_speed(cpose, 0,0,0)
        self.initial_z = 1.5
