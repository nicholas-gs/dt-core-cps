#!/usr/bin/env python3

import yaml
import rclpy

from functools import partial
from rclpy.node import Node
from builtin_interfaces.msg import Time
from std_msgs.msg import UInt8, Float32

from dt_interfaces_cps.msg import (
    AprilTagsWithInfos,
    Twist2DStamped,
    BoolStamped
)


MSG_TYPES = {
    "UInt8": UInt8,
    "Float32": Float32,
    "BoolStamped": BoolStamped,
    "Twist2DStamped": Twist2DStamped,
    "AprilTagsWithInfos": AprilTagsWithInfos,
}


class LogicGateNode(Node):

    VALID_GATE_TYPES = ["AND", "OR"]

    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.load_config_file(self.get_param_file_path())

        self.sub_list = list()
        self.pub_dict = dict()
        self.event_msg_dict = dict()
        self.event_trigger_dict = dict()
        self.last_published_msg = None

        for gate_name, gate_dict in self.gates_dict.items():
            output_topic_name = gate_dict["output_topic"]
            self.pub_dict[gate_name] = self.create_publisher(BoolStamped,
                output_topic_name, 1)

        for event_name, event_dict in self.events_dict.items():
            topic_name = event_dict["topic"]
            self.event_trigger_dict[event_name] = event_dict["trigger"]
            # Initialize local copy as None
            self.event_msg_dict[event_name] = None
            self.sub_list.append(
                self.create_subscription(
                    MSG_TYPES[event_dict["msg_type"]],
                    topic_name,
                    partial(self.bool_stamped_cb, event_name=event_name),
                    1
                )
            )

        self.get_logger().info("Intialized.")

    def get_param_file_path(self) -> str:
        """Get the `param_file_path` parameter from launch"""
        self.declare_parameter("param_file_path")
        return self.get_parameter("param_file_path").get_parameter_value()\
            .string_value

    def load_config_file(self, file_path: str):
        """Load the YAML configuration file and validate its contents."""
        configs = None
        with open(file_path, 'r') as f:
            configs = yaml.safe_load(f)

        self.gates_dict = configs['gates']
        self.events_dict = configs['events']

        # Validate the parameter file
        self._validate_gates(self.gates_dict)
        self._validate_events(self.events_dict)

    def _validate_gates(self, gates_dict):
        for _, gate_dict in gates_dict.items():
            gate_type = gate_dict["gate_type"]
            if gate_type not in self.VALID_GATE_TYPES:
                self.get_logger().fatal(f"Gate type {gate_type} is not valid")
                raise ValueError()

    def _validate_events(self, events_dict):
        for event_name, event_dict in events_dict.items():
            if "topic" not in event_dict:
                self.get_logger().fatal(
                    f"{self.get_name()} topic not defined for event {event_name}")
                raise ValueError()

    def publish(self, msg, gate_name):
        if msg is None:
            return
        self.pub_dict[gate_name].publish(msg)

    def get_output_msg(self, gate_name, inputs):
        bool_list = list()
        latest_time_stamp = Time(sec=0, nanosec=0)

        for event_name, event_msg in self.event_msg_dict.items():
            if event_name in inputs:
                if event_msg is None:
                    if "default" in self.events_dict[event_name]:
                        bool_list.append(self.events_dict[event_name]["default"])
                    else:
                        bool_list.append(False)
                else:
                    if "field" in self.events_dict[event_name]:
                        if ( getattr(event_msg, self.events_dict[event_name]["field"])
                            == self.event_trigger_dict[event_name]):
                            bool_list.append(True)
                        else:
                            bool_list.append(False)
                    else:
                        if event_msg.data == self.event_trigger_dict[event_name]:
                            bool_list.append(True)
                        else:
                            bool_list.append(False)

                    if LogicGateNode.after(
                        event_msg.header.stamp, latest_time_stamp
                    ):
                        latest_time_stamp = event_msg.header.stamp

        # Perform logic operation
        msg = BoolStamped()
        msg.header.stamp = latest_time_stamp

        gate = self.gates_dict.get(gate_name)
        gate_type = gate.get("gate_type")
        if gate_type == "AND":
            msg.data = all(bool_list)
        elif gate_type == "OR":
            msg.data = any(bool_list)

        return msg

    def bool_stamped_cb(self, msg, event_name: str):
        self.event_msg_dict[event_name] = msg
        for gate_name, gate_dict in self.gates_dict.items():
            inputs = gate_dict.get("inputs")
            for event_name in inputs:
                self.publish(self.get_output_msg(gate_name, inputs), gate_name)

    @staticmethod
    def after(t1: Time, t2: Time) -> bool:
        """Check if `t1` is at/after `t2`."""
        return t1.sec >= t2.sec and t1.nanosec >= t2.nanosec


def main(args=None):
    rclpy.init(args=args)
    node = LogicGateNode("logic_gate_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
