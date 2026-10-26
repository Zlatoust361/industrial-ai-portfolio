"""ROS2 launch file for vision_bridge.

Before starting the node, checks the Jetson's current `nvpmodel` power mode
and logs a warning (does not fail the launch) if it isn't one expected to
sustain stable detection FPS under continuous load -- see
`vision_bridge.power_mode`. `7W` is the Orin Nano's lowest power preset and
caps GPU/DLA clocks well below `15W`/`MAXN_SUPER`; it may still be a
deliberate choice on battery power, hence a warning rather than a hard
failure. Pass `check_power_mode:=false` to skip the check entirely (e.g.
off-device dev testing, where `nvpmodel` doesn't exist and the check is a
silent no-op anyway).

Usage:
    ros2 launch vision_bridge vision_bridge.launch.py \\
        image_topic:=/camera/image_raw text_query:="a photo of a defect"
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.launch_context import LaunchContext
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from vision_bridge.power_mode import is_power_mode_sufficient, query_current_power_mode


def _check_power_mode(context: LaunchContext, *args: object, **kwargs: object) -> list:
    """`OpaqueFunction` callback: warn (never fail launch) if the power mode is insufficient."""
    if LaunchConfiguration("check_power_mode").perform(context).lower() != "true":
        return []

    mode_name = query_current_power_mode()
    if mode_name is None:
        # nvpmodel isn't present/runnable -- not on a Jetson, nothing to check.
        return []
    if not is_power_mode_sufficient(mode_name):
        print(
            f"[vision_bridge launch] WARNING: current nvpmodel power mode is "
            f"'{mode_name}', not 15W/MAXN_SUPER. 7W may not sustain stable "
            "detection FPS under continuous load. Switch with "
            "'sudo nvpmodel -m 1' (15W) or 'sudo nvpmodel -m 0' (MAXN Super), "
            "or pass check_power_mode:=false to silence this check."
        )
    return []


def generate_launch_description() -> LaunchDescription:
    """Build the vision_bridge launch description."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "check_power_mode",
                default_value="true",
                description="Warn if nvpmodel isn't 15W/MAXN_SUPER before starting.",
            ),
            DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("text_query", default_value="object"),
            DeclareLaunchArgument("text_query_topic", default_value="~/text_query"),
            DeclareLaunchArgument("detections_topic", default_value="~/detections"),
            OpaqueFunction(function=_check_power_mode),
            Node(
                package="vision_bridge",
                executable="vision_bridge_node",
                name="vision_bridge",
                output="screen",
                parameters=[
                    {
                        "image_topic": LaunchConfiguration("image_topic"),
                        "text_query": LaunchConfiguration("text_query"),
                        "text_query_topic": LaunchConfiguration("text_query_topic"),
                        "detections_topic": LaunchConfiguration("detections_topic"),
                    }
                ],
            ),
        ]
    )
