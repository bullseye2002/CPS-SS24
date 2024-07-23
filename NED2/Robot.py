from pyniryo2 import *
import pyniryo
import logging


class Robot:

    def __init__(self, simulation=False):
        self.__robot = None
        self.__hotspot_mode = "10.10.10.10"
        self.__wifi_mode = "192.168.0.140"
        self.__simulation_mode = simulation
        self.__observation_pose = PoseObject(
            x=0.21, y=0., z=0.2,
            roll=0.0, pitch=2., yaw=0.0,
        )
        self.__logger = logging.getLogger(__name__)
        # Configure logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler("robot_logs.log"),
                                logging.StreamHandler()
                            ])
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

        self.__logger.info(f"Connecting to robot at IP {robot_ip_address} in {'simulation' if self.__simulation_mode else 'wifi'} mode")

        self.__robot = NiryoRobot(robot_ip_address)

        if not self.__simulation_mode:
            # Calibrate robot if robot needs calibration
            self.__robot.arm.calibrate_auto()

            # Equip tool
            self.__robot.tool.update_tool()

    #TODO: Könnte man entfernen, wird nirgends verwendet
    def get_robot(self) -> NiryoRobot:
        return self.__robot

    #TODO: wird nur von move_to_observation_pose aufgerufen
    # Funktion kann gestrichen und in der gesagten Funktion übernommen werden
    def __move_pose(self, pose: PoseObject):
        self.__robot.arm.move_pose(pose)

    def move_xy(self, x, y, z=0.035):
        self.__robot.arm.move_pose(PoseObject(
            x=x, y=y, z=z,
            roll=-0.142, pitch=1.57, yaw=0.
        ))

    def move_to_observation_pose(self):
        self.__move_pose(self.__observation_pose)

    def take_image(self):
        img_compressed = self.__robot.vision.get_img_compressed()
        camera_info = self.__robot.vision.get_camera_intrinsics()
        image = pyniryo.uncompress_image(img_compressed)
        image = pyniryo.undistort_image(image, camera_info.intrinsics, camera_info.distortion)

        return image

    def move_to_home_pose(self):
        self.__robot.arm.move_to_home_pose()

    def disconnect(self):
        # Ending
        self.__robot.arm.go_to_sleep()
        # Releasing connection
        self.__robot.end()