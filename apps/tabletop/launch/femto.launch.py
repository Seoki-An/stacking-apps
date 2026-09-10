"""Explicit driver bringup for one calibrated FEMTO; capture stays in the app."""

from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _setup(context):
    from stacking_tabletop.config.femto import load_femto_profile
    from stacking_tabletop.acquisition.femto import FemtoCalibration
    profile = load_femto_profile(LaunchConfiguration('config').perform(context), LaunchConfiguration('camera').perform(context))
    FemtoCalibration(profile)  # Refuse inconsistent files before launching hardware.
    driver = Path(get_package_share_directory('orbbec_camera')) / 'launch/femto_bolt.launch.py'
    return [IncludeLaunchDescription(PythonLaunchDescriptionSource(str(driver)), launch_arguments={
        'serial_number': profile.serial, 'camera_name': profile.namespace.strip('/'),
        'enable_point_cloud': 'true', 'enable_color': 'true', 'enable_ir': 'false',
        'depth_width': '1024', 'depth_height': '1024', 'depth_fps': '15',
        'ir_width': '1024', 'ir_height': '1024', 'ir_fps': '15',
    }.items())]


def generate_launch_description():
    config = Path(__file__).resolve().parents[1] / 'config/femto.yml'
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=str(config)),
        DeclareLaunchArgument('camera', default_value='primary', choices=['primary', 'secondary']),
        OpaqueFunction(function=_setup),
    ])
