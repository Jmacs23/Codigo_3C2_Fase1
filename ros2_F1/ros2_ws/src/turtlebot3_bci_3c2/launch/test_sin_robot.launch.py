#!/usr/bin/env python3
"""
test_sin_robot.launch.py --- Arranque sin robot fisico
Paquete: turtlebot3_bci_3c2

Levanta solo los nodos que no necesitan hardware, para probar la cadena de
comunicacion con la maquina Windows antes de tocar el robot.

    ros2 launch turtlebot3_bci_3c2 test_sin_robot.launch.py

Que se puede comprobar con esto:
  - que bci_node recibe los mensajes de Windows
  - que mission_node produce comandos coherentes
  - que interface_node devuelve el estado a la interfaz

Que NO se puede comprobar: nada que dependa del LiDAR o de la odometria.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("turtlebot3_bci_3c2")
    params = os.path.join(pkg, "config", "params.yaml")
    windows_ip = LaunchConfiguration("windows_ip")

    return LaunchDescription([
        DeclareLaunchArgument("windows_ip", default_value="127.0.0.1"),
        Node(package="turtlebot3_bci_3c2", executable="bci_node.py",
             name="bci_node", parameters=[params], output="screen",
             emulate_tty=True),
        Node(package="turtlebot3_bci_3c2", executable="mission_node.py",
             name="mission_node", parameters=[params], output="screen",
             emulate_tty=True),
        Node(package="turtlebot3_bci_3c2", executable="interface_node.py",
             name="interface_node",
             parameters=[params, {"host": windows_ip}], output="screen"),
    ])
