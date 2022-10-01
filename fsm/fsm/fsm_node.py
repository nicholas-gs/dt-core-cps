#!/usr/bin/env python3

import copy
import yaml
import rclpy

from typing import Optional
from functools import partial
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, 
    QoSDurabilityPolicy
)

from dt_interfaces_cps.msg import (
    FSMState,
    BoolStamped
)
from dt_interfaces_cps.srv import SetFSMState


MSG_TYPES = {
    "BoolStamped": BoolStamped,
}


class FSMNode(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.load_config_file(self.get_param_file_path())

        # Setup intial state
        self.state_msg = FSMState()
        self.state_msg.state = self._initial_state
        self.state_msg.header.stamp = self.get_clock().now().to_msg()

        # `latching_qos` to mimic the `latch=True` in rospy.Publisher(...).
        # But will need to test if it actually works :/
        latching_qos = QoSProfile(depth=1,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL)
        self.pub_state = self.create_publisher(
            FSMState, "~/mode", latching_qos)

        self.srv_state = self.create_service(
            SetFSMState, "~/set_state", self.srv_set_state_cb)

        self.active_nodes = None
        # For each node defined in the param file, create a publisher for its
        # defined topic
        self.pub_dict = {
            node_name : self.create_publisher(BoolStamped, topic_name, latching_qos)
                for node_name, topic_name in self.nodes.items()
        }

        self.sub_list = list()
        self.event_trigger_dict = dict()
        for event_name, event_dict in self.events_dict.items():
            topic_name = event_dict['topic']
            msg_type = event_dict['msg_type']
            self.event_trigger_dict[event_name] = event_dict["trigger"]
            self.sub_list.append(self.create_subscription(
                MSG_TYPES[msg_type], 
                topic_name,
                partial(self.cbEvent, event_name=event_name),
                1)
            )

        self.get_logger().info("Intialized.")

        # Publish the initial state
        self.publish()

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
        
        self._initial_state = configs.get('initial_state', '')
        self.states_dict = configs['states']
        self.global_transitions_dict = configs['global_transitions']
        self.events_dict = configs['events']
        self.nodes = configs['nodes']

        # Validate the param file
        self._validate_states(self.states_dict)
        self._validate_global_transitions(
            self.global_transitions_dict, list(self.states_dict.keys()))
        self._validate_events(self.events_dict)

    def _validate_states(self, states_dict):
        """Validate the `states` section of the YAML configuration file."""
        valid_states = list(states_dict.keys())
        for state, state_dict in states_dict.items():
            # Validate the existence of all reachable states
            transitions_dict = state_dict.get("transitions")
            if transitions_dict is None:
                continue
            else:
                for transition, next_state in transitions_dict.items():
                    if next_state not in valid_states:
                        self.get_logger().fatal(f"{next_state} not a valid"\
                            f" state, (From {state} with event {transition}")
                        raise ValueError()

    def _validate_global_transitions(self, global_transitions, valid_states):
        """Validate the `global_transitions` section of the YAML configuration file."""
        for event_name, state_name in global_transitions.items():
            if state_name not in valid_states:
                self.get_logger().fatal(f"State {state_name} is not valid."\
                    f" (From global_transitions of {event_name}")
                raise ValueError()

    def _validate_events(self, events_dict):
        """Validate the `events` section of the YAML configuration file."""
        expected_fields = ['topic', 'msg_type', 'trigger']
        for event_name, event_dict in events_dict.items():
            fields = list(event_dict.keys())
            missing_fields = list(
                filter(lambda ef: ef not in fields, expected_fields))
            if len(missing_fields) > 0:
                self.get_logger().fatal(f"Event {event_name}"\
                    f" missing {missing_fields} definitions")
                raise ValueError()

    def _get_next_state(self, state_name: str, event_name: str) -> Optional[str]:
        if not self.is_valid_state(state_name):
            self.get_logger().warn(f"{state_name} not defined. Treating as terminal")
            return None
        # `state` transitions overwrites `global_transitions`
        state_dict = self.states_dict.get(state_name)
        if "transitions" in state_dict:
            next_state = state_dict["transitions"].get(event_name)
        else:
            next_state = None
        
        # If cannot find transition, then look for it in `global_transitions`
        if next_state is None:
            next_state = self.global_transitions_dict.get(event_name)
        
        return next_state

    def _get_active_nodes_of_state(self, state_name: str):
        active_nodes = self.states_dict[state_name].get("active_nodes", [])
        if active_nodes is None:
            self.get_logger().warn(f"No active nodes defined for {state_name}.")
        return active_nodes            

    def publish(self):
        self.publish_bools()
        self.publish_state()

    def is_valid_state(self, state: str) -> bool:
        return state in self.states_dict
    
    def srv_set_state_cb(self, request, response):
        req_state = request.state
        if self.is_valid_state(req_state):
            self.state_msg.header.stamp = self.get_clock().now().to_msg()
            self.state_msg.state = req_state
            self.publish()
        else:
            self.get_logger().warn(f"{req_state} is not a valid state.")
        return response

    def publish_state(self):
        self.pub_state.publish(self.state_msg)
    
    def publish_bools(self):
        active_nodes = self._get_active_nodes_of_state(self.state_msg.state)
        for node_name, node_pub in self.pub_dict.items():
            msg = BoolStamped()
            msg.header.stamp = self.state_msg.header.stamp
            msg.data = bool(node_name in active_nodes)
            node_state = "ON" if msg.data else "OFF"
            if self.active_nodes is not None:
                if (node_name in active_nodes) == (node_name in self.active_nodes):
                    continue
            node_pub.publish(msg)
        self.active_nodes = copy.deepcopy(active_nodes)

    def cbEvent(self, msg: BoolStamped, event_name: str):
        """Callback function for the subscription from various nodes"""
        if msg.data == self.event_trigger_dict[event_name]:
            self.state_msg.header.stamp = msg.header.stamp
            next_state = self._get_next_state(self.state_msg.state, event_name)
            if next_state is not None:
                self.state_msg.state = next_state
                self.publish()


def main(args=None):
    rclpy.init(args=args)
    node = FSMNode("fsm_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
