from pyniryo2 import *
import pyniryo


class Robot:

    def __init__(self, simulation=False):
        self.__robot = None
        self.__hotspot_mode = "10.10.10.10"
        self.__wifi_mode = "192.168.0.140"
        self.__simulation_mode = simulation
        self.__observation_pose = PoseObject(
            x=0.17, y=0., z=0.35,
            roll=0.0, pitch=1.57, yaw=0.0,
        )
        super().__init__()

    def connect(self):
        if self.__simulation_mode:
            robot_ip_address, workspace_name = "172.16.196.129", "gazebo_1"
        else:
            robot_ip_address, workspace_name = self.__wifi_mode, "cps_praktikum"

        # -- Can Change these variables
        grid_dimension = (3, 3)  # conditioning grid dimension
        vision_process_on_robot = False  # boolean to indicate if the image processing happens on the Robot
        display_stream = True  # Only used if vision on computer

        self.__robot = NiryoRobot(robot_ip_address)

        if not self.__simulation_mode:
            # Calibrate robot if robot needs calibration
            self.__robot.arm.calibrate_auto()

            # Equip tool
            self.__robot.tool.update_tool()

    def get_robot(self) -> NiryoRobot:
        return self.__robot

    def move_pose(self, pose: PoseObject):
        self.__robot.arm.move_pose(pose)

    def move_xy(self, x, y):
        self.__robot.arm.move_pose(PoseObject(
            x=x, y=y, z=0.1,
            roll=-0.142, pitch=1.57, yaw=0.
        ))

    def move_to_observation_pose(self):
        self.move_pose(self.__observation_pose)

    def take_image(self):
        img_compressed = self.__robot.vision.get_img_compressed()
        camera_info = self.__robot.vision.get_camera_intrinsics()
        image = pyniryo.uncompress_image(img_compressed)
        image = pyniryo.undistort_image(image, camera_info.intrinsics, camera_info.distortion)

        return image

    def disconnect(self):
        # Ending
        self.__robot.arm.go_to_sleep()
        # Releasing connection
        self.__robot.end()