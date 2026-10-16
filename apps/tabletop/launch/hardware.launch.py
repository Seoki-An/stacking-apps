"""Start arm/gripper, FEMTO, and the independent perception service."""

from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


_COMMISSIONED_ROBOT_IP = "192.168.56.101"
_COMMISSIONED_KINEMATICS = Path.home() / "ros2_ws" / "ur5e_calibration.yaml"


def _ur5e_launch(name: str) -> PythonLaunchDescriptionSource:
    path = Path(get_package_share_directory("ur5e_admittance")) / "launch" / name
    return PythonLaunchDescriptionSource(str(path))


def generate_launch_description() -> LaunchDescription:
    robot_ip = LaunchConfiguration("robot_ip")
    kinematics = LaunchConfiguration("kinematics_params_file")
    gripper_device = LaunchConfiguration("gripper_device")
    start_gripper = LaunchConfiguration("start_gripper")

    arm = IncludeLaunchDescription(
        _ur5e_launch("bringup.launch.py"),
        launch_arguments={
            "robot_ip": robot_ip,
            "kinematics_params_file": kinematics,
            "use_mock_hardware": LaunchConfiguration("use_mock_hardware"),
            "headless_mode": LaunchConfiguration("headless_mode"),
            "launch_rviz": LaunchConfiguration("launch_rviz"),
        }.items(),
    )
    gripper = IncludeLaunchDescription(
        _ur5e_launch("gripper_bringup.launch.py"),
        launch_arguments={
            "device": gripper_device,
            "hold_preload_counts": LaunchConfiguration("hold_preload_counts"),
            "hold_recovery_max_opening_counts": LaunchConfiguration(
                "hold_recovery_max_opening_counts"
            ),
        }.items(),
        condition=IfCondition(start_gripper),
    )
    repository = Path(__file__).resolve().parents[3]
    femto = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(Path(__file__).with_name('femto.launch.py'))),
        launch_arguments={'camera': LaunchConfiguration('femto_camera')}.items(),
        condition=IfCondition(LaunchConfiguration('start_femto')),
    )
    perception = ExecuteProcess(
        cmd=[LaunchConfiguration('perception_python'), '-u', '-m',
             'stacking_tabletop.perception.service', '--repository', str(repository),
             '--wrist-camera', LaunchConfiguration('start_wrist_camera')],
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_perception')),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_ip",
                default_value=_COMMISSIONED_ROBOT_IP,
                description="IP address of the UR5e robot",
            ),
            DeclareLaunchArgument(
                "kinematics_params_file",
                default_value=str(_COMMISSIONED_KINEMATICS),
                description="UR factory calibration YAML",
            ),
            DeclareLaunchArgument("use_mock_hardware", default_value="false"),
            DeclareLaunchArgument("headless_mode", default_value="false"),
            DeclareLaunchArgument("launch_rviz", default_value="false"),
            DeclareLaunchArgument("start_gripper", default_value="true"),
            DeclareLaunchArgument('start_femto', default_value='true'),
            DeclareLaunchArgument('femto_camera', default_value='primary', choices=['primary', 'secondary']),
            DeclareLaunchArgument('start_perception', default_value='true'),
            DeclareLaunchArgument('start_wrist_camera', default_value='true'),
            DeclareLaunchArgument('perception_python', default_value=sys.executable),
            DeclareLaunchArgument("gripper_device", default_value="/dev/ttyUSB0"),
            DeclareLaunchArgument("hold_preload_counts", default_value="10"),
            DeclareLaunchArgument(
                "hold_recovery_max_opening_counts", default_value="50"
            ),
            arm,
            gripper,
            femto,
            perception,
        ]
    )
