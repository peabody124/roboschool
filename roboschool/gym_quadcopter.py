from roboschool.gym_forward_walker import RoboschoolForwardWalker
from roboschool.gym_urdf_robot_env import RoboschoolUrdfEnv
from roboschool.multiplayer import SharedMemoryClientEnv
from roboschool.scene_abstract import cpp_household
from roboschool.scene_stadium import SinglePlayerStadiumScene
import numpy as np

class RoboschoolQuadcopterBase(SharedMemoryClientEnv, RoboschoolUrdfEnv):
    random_yaw = False

    def __init__(self):
        RoboschoolUrdfEnv.__init__(self,
            "quadcopter_description/urdf/quadcopter-v1.urdf",
            "center",
            action_dim=4, obs_dim=12,
            fixed_base=False,
            self_collision=False)

        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
        self.camera_x = 0
        self.camera_y = 4.3
        self.camera_z = 45.0
        self.camera_follow = 0

        self.count = 0

    MASS = 2
    GRAV = 9.8
    TORQUE_SCALE = 0.05

    def create_single_player_scene(self):
        return SinglePlayerStadiumScene(gravity=self.GRAV, timestep=0.0165/8, frame_skip=8)   # 8 instead of 4 here

    def apply_action(self, a):
        assert( np.isfinite(a).all() )

        # the first three parameters are the roll, pitch and yaw signals to command. 
        # we apply a scale to go from teh command signals (0,1) to the torque range
        # which will then be filtered by the moment of inertia of the fram
        self.cpp_robot.apply_external_torque(self.TORQUE_SCALE * a[0], self.TORQUE_SCALE * a[1], self.TORQUE_SCALE * a[2])

        # the third parameter is throttle (from 0, 1) with full
        # throttle corresponding to 2 masses worth of force
        self.cpp_robot.apply_external_force(0,0,self.GRAV*self.MASS*2*a[3])


    def robot_specific_reset(self):
        self.scene.actor_introduce(self)
        self.set_initial_orientation(yaw_center=0, yaw_random_spread=np.pi)
        self.base = self.parts["center"]

    def move_robot(self, init_x, init_y, init_z):
        "Used by multiplayer stadium to move sideways, to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        pose.move_xyz(init_x, init_y, init_z)  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        self.cpp_robot.set_pose(pose)
        self.start_pos_x, self.start_pos_y, self.start_pos_z = init_x, init_y, init_z

    def set_initial_orientation(self, yaw_center, yaw_random_spread):
        cpose = cpp_household.Pose()
        if not self.random_yaw:
            yaw = yaw_center
        else:
            yaw = yaw_center + self.np_random.uniform(low=-yaw_random_spread, high=yaw_random_spread)

        cpose.set_xyz(self.start_pos_x, self.start_pos_y, self.start_pos_z + 1.5)
        cpose.set_rpy(0, 0, yaw)  # just face random direction, but stay straight otherwise
        self.cpp_robot.set_pose_and_speed(cpose, 0,0,0)
        self.initial_z = 1.5

    def calc_state(self):
        # state is just concatenation of position and orientation
        # can later augment with additional information like velocities
        # and rotational moments. can additional corrupt with noise.

        # TODO: angular_speed is in world frame, need to convert to body frame
        state = np.concatenate((self.base.pose().xyz(), self.base.pose().rpy(), self.base.speed(), self.base.angular_speed()))

        return state

    def _step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.apply_action(a)
            self.scene.global_step()

        state = self.calc_state()  # also calculates self.joints_at_limit

        alive = float(self.alive_bonus())   # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        self.rewards = [
            alive,
            ]

        self.frame  += 1
        if (done and not self.done) or self.frame==self.spec.timestep_limit:
            self.episode_over(self.frame)
        self.done   += done   # 2 == 1+True
        self.reward += sum(self.rewards)
        self.HUD(state, a, done)
        return state, sum(self.rewards), bool(done), {}

    def episode_over(self, frames):
        pass

    def camera_adjust(self):
        #self.camera_dramatic()
        self.camera_simple_follow()

    def camera_simple_follow(self):
        x, y, z = self.base.pose().xyz()
        self.camera_x = 0.98*self.camera_x + (1-0.98)*x
        self.camera.move_and_look_at(self.camera_x, y-5.0, 4, x, y, z)

    def camera_dramatic(self):
        pose = self.base.pose()
        speed = self.base.speed()
        x, y, z = pose.xyz()
        if 1:
            camx, camy, camz = speed[0], speed[1], 2.2
        else:
            camx, camy, camz = self.walk_target_x - x, self.walk_target_y - y, 2.2

        n = np.linalg.norm([camx, camy])
        if n > 2.0 and self.frame > 50:
            self.camera_follow = 1
        if n < 0.5:
            self.camera_follow = 0
        if self.camera_follow:
            camx /= 0.1 + n
            camx *= 2.2
            camy /= 0.1 + n
            camy *= 2.8
            if self.frame < 1000:
                camx *= -1
                camy *= -1
            camx += x
            camy += y
            camz  = 1.8
        else:
            camx = x
            camy = y + 4.3
            camz = 2.2
        #print("%05i" % self.frame, self.camera_follow, camy)
        smoothness = 0.97
        self.camera_x = smoothness*self.camera_x + (1-smoothness)*camx
        self.camera_y = smoothness*self.camera_y + (1-smoothness)*camy
        self.camera_z = smoothness*self.camera_z + (1-smoothness)*camz
        self.camera.move_and_look_at(self.camera_x, self.camera_y, self.camera_z, x, y, 0.6)


class RoboschoolQuadcopterHover(RoboschoolQuadcopterBase):

    def __init__(self):
        RoboschoolQuadcopterBase.__init__(self)

    def alive_bonus(self):

        state = self.calc_state()
        z = state[2]

        if z < 1 or z > 5:
            # quit if it is flying away or crashing
            #print("Flying away")
            return -1

        desired_position = np.array([0,0,2.5])
        desired_rpy = np.array([0,0,0])
        desired_speed = np.array([0,0,0])
        desired_angular_speed = np.array([0,0,0])

        desired_state = np.concatenate((desired_position, desired_rpy, desired_speed, desired_angular_speed))

        pos_weight = 1
        rpy_weight = 1
        speed_weight = 1
        angular_speed_weight = 1

        desired_weights = np.array([pos_weight, pos_weight, pos_weight,
            rpy_weight, rpy_weight, rpy_weight,
            speed_weight, speed_weight, speed_weight,
            angular_speed_weight, angular_speed_weight, angular_speed_weight])

        # compute the squared error for each state and then the weighted sum of them
        state_error_sq = np.square(state - desired_state)
        weighted_error = np.sum(np.multiply(desired_weights, state_error_sq))

        reward = 100 - weighted_error

        #if reward < 0:
        #    print("Far from desired state")

        return reward