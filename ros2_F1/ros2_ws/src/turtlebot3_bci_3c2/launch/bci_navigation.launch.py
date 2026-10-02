#!/usr/bin/env python3
"""
bci_navigation.launch.py --- Arranque del sistema completo
Paquete: turtlebot3_bci_3c2

===========================================================================
AVISO: no verificado por ejecucion (ROS2 no disponible al escribirlo).
===========================================================================

USO
---
    ros2 launch turtlebot3_bci_3c2 bci_navigation.launch.py

    # Con la maquina Windows en otra IP:
    ros2 launch turtlebot3_bci_3c2 bci_navigation.launch.py \
        windows_ip:=192.168.1.50

    # Activando la ganancia variable (Fase 2):
    ros2 launch turtlebot3_bci_3c2 bci_navigation.launch.py \
        usar_lambda_bci:=true

    # Sin grabar rosbag:
    ros2 launch turtlebot3_bci_3c2 bci_navigation.launch.py \
        grabar:=false

ORDEN DE ARRANQUE
-----------------
safety_node arranca PRIMERO y con un retardo cero. Los demas esperan.

La razon es que si el robot ya esta encendido y hay un obstaculo cerca,
queremos proteccion desde el primer instante, antes de que nada pueda
mandarle un comando de avance.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            TimerAction, LogInfo)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("turtlebot3_bci_3c2")
    params = os.path.join(pkg, "config", "params.yaml")

    # --- Argumentos ---
    windows_ip = LaunchConfiguration("windows_ip")
    usar_lambda = LaunchConfiguration("usar_lambda_bci")
    grabar = LaunchConfiguration("grabar")

    args = [
        DeclareLaunchArgument(
            "windows_ip", default_value="127.0.0.1",
            description="IP de la maquina Windows con PsychoPy y el g.USBamp"),
        DeclareLaunchArgument(
            "usar_lambda_bci", default_value="false",
            description="Ganancia de control ponderada por lambda_bci. "
                        "false = Fase 1, true = Fase 2"),
        DeclareLaunchArgument(
            "grabar", default_value="true",
            description="Grabar rosbag del trial"),
    ]

    # --- Seguridad: arranca primero, sin retardo ---
    safety = Node(
        package="turtlebot3_bci_3c2",
        executable="safety_node.py",
        name="safety_node",
        parameters=[params],
        output="screen",
        emulate_tty=True,
    )

    # --- El resto de nodos, tras un breve retardo ---
    resto = TimerAction(
        period=1.0,
        actions=[
            Node(
                package="turtlebot3_bci_3c2",
                executable="bci_node.py",
                name="bci_node",
                parameters=[params],
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="turtlebot3_bci_3c2",
                executable="mission_node.py",
                name="mission_node",
                parameters=[params],
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="turtlebot3_bci_3c2",
                executable="ctrl_node.py",
                name="ctrl_node",
                parameters=[params, {"usar_lambda_bci": usar_lambda}],
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="turtlebot3_bci_3c2",
                executable="interface_node.py",
                name="interface_node",
                parameters=[params, {"host": windows_ip}],
                output="screen",
            ),
            Node(
                package="turtlebot3_bci_3c2",
                executable="video_publisher.py",
                name="video_publisher",
                parameters=[params, {"host": windows_ip}],
                output="screen",
            ),
        ],
    )

    # --- Grabacion ---
    #
    # Se graban solo los topics necesarios para reanalizar un trial. Grabar
    # el video multiplicaria el tamano por cien y no aporta al analisis: la
    # deteccion ya ocurrio y sus resultados estan en /bci/objetos.
    rosbag = ExecuteProcess(
        cmd=[
            "ros2", "bag", "record",
            "/tasm/state", "/bci/command", "/cmd_vel", "/scan", "/odom",
            "/bci/emergencia", "/bci/etapa", "/bci/objetos",
            "-o", "trial",
        ],
        output="screen",
        condition=IfCondition(grabar),
    )

    return LaunchDescription(args + [
        LogInfo(msg="Arrancando safety_node primero..."),
        safety,
        resto,
        TimerAction(period=2.0, actions=[rosbag]),
    ])
