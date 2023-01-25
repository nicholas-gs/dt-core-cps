#!/usr/bin/env python3

import json
import rclpy
import numpy as np

from functools import partial

from rclpy.node import Node
from rclpy.time import Time, Duration
from rcl_interfaces.msg import (
    ParameterType,
    FloatingPointRange,
    ParameterDescriptor,
    SetParametersResult
)

from lane_control.controller import LaneController
from dt_interfaces_cps.msg import (
    LanePose,
    Twist2DStamped,
    StopLineReading,
    WheelsCmdStamped
)


class LaneControllerNode(Node):
    """Computes control action.
    The node compute the commands in form of linear and angular velocities,
    by processing the estimate error in lateral deviationa and heading.
    The configuration parameters can be changed dynamically while the node is
    running.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that
            ROS will use
    Configuration:
        ~v_bar (:obj:`float`): Nominal velocity in m/s
        ~k_d (:obj:`float`): Proportional term for lateral deviation
        ~k_theta (:obj:`float`): Proportional term for heading deviation
        ~k_Id (:obj:`float`): integral term for lateral deviation
        ~k_Iphi (:obj:`float`): integral term for lateral deviation
        ~d_thres (:obj:`float`): Maximum value for lateral error
        ~theta_thres (:obj:`float`): Maximum value for heading error
        ~d_offset (:obj:`float`): Goal offset from center of the lane
        ~integral_bounds (:obj:`dict`): Bounds for integral term
        ~d_resolution (:obj:`float`): Resolution of lateral position estimate
        ~phi_resolution (:obj:`float`): Resolution of heading estimate
        ~omega_ff (:obj:`float`): Feedforward part of controller
        ~verbose (:obj:`bool`): Verbosity level (0,1,2)
        ~stop_line_slowdown (:obj:`dict`): Start and end distances for
            slowdown at stop lines
    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the
            lane filter
        ~intersection_navigation_pose (:obj:`LanePose`): The lane pose estimate
            from intersection navigation
        ~wheels_cmd_executed (:obj:`WheelsCmdStamped`): Confirmation that the
            control action was executed
        ~stop_line_reading (:obj:`StopLineReading`): Distance from stopline,
            to reduce speed
        ~obstacle_distance_reading (:obj:`stop_line_reading`): Distance from
            obstacle virtual stopline, to reduce speed
    """

    # List of all the ROS parameter names
    ROS_PARAM_NAMES = (
        'v_bar',
        'k_d',
        'k_theta',
        'k_Id',
        'k_Iphi',
        'theta_thres',
        'd_thres',
        'd_offset',
        'omega_ff',
        'integral_bounds.d.top',
        'integral_bounds.d.bot',
        'integral_bounds.phi.top',
        'integral_bounds.phi.bot',
        'd_resolution',
        'phi_resolution',
        'stop_line_slowdown.start',
        'stop_line_slowdown.end',
        'verbose'
    )

    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.load_launch_params()
        self.add_on_set_parameters_callback(self.param_update_cb)

        self.controller = LaneController(self.params)

        # Initialize variables
        self.fsm_state = None
        self.received_wheels_cmd_executed = False
        self.wheels_cmd_executed = WheelsCmdStamped()
        self.pose_msg = LanePose()
        self.pose_initialized = False
        self.pose_msg_dict = dict()
        self.last_s: Time = None
        self.stop_line_distance = None
        self.stop_line_detected = False
        self.at_stop_line = False
        self.obstacle_stop_line_distance = None
        self.obstacle_stop_line_detected = False
        self.at_obstacle_stop_line = False

        self.current_pose_source = 'lane_filter'

        # Create publishers
        self.pub_car_cmd = self.create_publisher(
            Twist2DStamped,
            '~/car_cmd',
            1)

        # Create Subscriptions
        self.sub_lane_reading = self.create_subscription(
            LanePose,
            '~/lane_pose',
            partial(self.all_poses_cb, pose_source='lane_filter'),
            1)
        self.sub_intersection_navigation_pose = self.create_subscription(
            LanePose,
            "~/intersection_navigation_pose",
            partial(self.all_poses_cb, pose_source='intersection_navigation'),
            1)
        self.sub_wheels_cmd = self.create_subscription(
            WheelsCmdStamped,
            "~/wheels_cmd",
            self.wheels_cmd_executed_cb,
            1)
        self.sub_stop_line = self.create_subscription(
            StopLineReading,
            "~/stop_line_reading",
            self.stop_line_reading_cb,
            1)
        self.sub_obstacle_stop_line = self.create_subscription(
            StopLineReading,
            "~/obstacle_distance_reading",
            self.obstacle_stop_line_reading_cb,
            1)

        dump = json.dumps(self.params, sort_keys=True, indent=2)
        self.get_logger().info(f"Initialised with {dump}")

    def load_launch_params(self) -> None:
        """Retrieve and store all the launch parameters."""
        self.params = {
            '~integral_bounds' : {'d' : {}, 'phi' : {}},
            '~stop_line_slowdown' : {}
        }
        self.declare_parameters(namespace='',
            parameters=[
                ('v_bar', None, ParameterDescriptor(
                    name='v_bar',
                    type=ParameterType.PARAMETER_DOUBLE,
                    floating_point_range=[FloatingPointRange(from_value=0.0,
                        to_value=5.0)])),
                ('k_d', None, ParameterDescriptor(
                    name='k_d',
                    type=ParameterType.PARAMETER_DOUBLE,
                    floating_point_range=[FloatingPointRange(from_value=-100.0,
                        to_value=100.0)])),
                ('k_theta', None, ParameterDescriptor(
                    name='k_theta',
                    type=ParameterType.PARAMETER_DOUBLE,
                    floating_point_range=[FloatingPointRange(from_value=-100.0,
                        to_value=100.0)])),
                ('k_Id', None, ParameterDescriptor(
                    name='k_Id',
                    type=ParameterType.PARAMETER_DOUBLE,
                    floating_point_range=[FloatingPointRange(from_value=-100.0,
                        to_value=100.0)])),
                ('k_Iphi', None, ParameterDescriptor(
                    name='k_Iphi',
                    type=ParameterType.PARAMETER_DOUBLE,
                    floating_point_range=[FloatingPointRange(from_value=-100.0,
                        to_value=100.0)])),
                ('theta_thres', None, ParameterDescriptor(
                    name='theta_thres',
                    type=ParameterType.PARAMETER_DOUBLE)),
                ('d_thres', None, ParameterDescriptor(
                    name='d_thres',
                    type=ParameterType.PARAMETER_DOUBLE)),
                ('d_offset', None, ParameterDescriptor(
                    name='d_offset',
                    type=ParameterType.PARAMETER_DOUBLE)),
                ('omega_ff', None, ParameterDescriptor(
                    name='omega_ff',
                    type=ParameterType.PARAMETER_DOUBLE)),
                ('integral_bounds.d.top', None, ParameterDescriptor(
                    name='integral_bounds.d.top',
                    type=ParameterType.PARAMETER_DOUBLE)),
                ('integral_bounds.d.bot', None, ParameterDescriptor(
                    name='integral_bounds.d.bot',
                    type=ParameterType.PARAMETER_DOUBLE)),
                ('integral_bounds.phi.top', None, ParameterDescriptor(
                    name='integral_bounds.phi.top',
                    type=ParameterType.PARAMETER_DOUBLE)),
                ('integral_bounds.phi.bot', None, ParameterDescriptor(
                    name='integral_bounds.phi.bot',
                    type=ParameterType.PARAMETER_DOUBLE)),
                ('d_resolution', None, ParameterDescriptor(
                    name='d_resolution',
                    type=ParameterType.PARAMETER_DOUBLE)),
                ('phi_resolution', None, ParameterDescriptor(
                    name='phi_resolution',
                    type=ParameterType.PARAMETER_DOUBLE)),
                ('stop_line_slowdown.start', None, ParameterDescriptor(
                    name='stop_line_slowdown.start',
                    type=ParameterType.PARAMETER_DOUBLE)),
                ('stop_line_slowdown.end', None, ParameterDescriptor(
                    name='stop_line_slowdown.end',
                    type=ParameterType.PARAMETER_DOUBLE))
            ])
        self.params['~v_bar'] = self.get_parameter('v_bar')\
            .get_parameter_value().double_value
        self.params['~k_d'] = self.get_parameter('k_d')\
            .get_parameter_value().double_value
        self.params['~k_theta'] = self.get_parameter('k_theta')\
            .get_parameter_value().double_value
        self.params['~k_Id'] = self.get_parameter('k_Id')\
            .get_parameter_value().double_value
        self.params['~k_Iphi'] = self.get_parameter('k_Iphi')\
            .get_parameter_value().double_value
        self.params['~theta_thres'] = self.get_parameter('theta_thres')\
            .get_parameter_value().double_value
        self.params['~d_thres'] = self.get_parameter('d_thres')\
            .get_parameter_value().double_value
        self.params['~d_offset'] = self.get_parameter('d_offset')\
            .get_parameter_value().double_value
        self.params['~omega_ff'] = self.get_parameter('omega_ff')\
            .get_parameter_value().double_value

        self.params['~integral_bounds']['d']['top'] = self.get_parameter(
            'integral_bounds.d.top').get_parameter_value().double_value
        self.params['~integral_bounds']['d']['bot'] = self.get_parameter(
            'integral_bounds.d.bot').get_parameter_value().double_value
        self.params['~integral_bounds']['phi']['top'] = self.get_parameter(
            'integral_bounds.phi.top').get_parameter_value().double_value
        self.params['~integral_bounds']['phi']['bot'] = self.get_parameter(
            'integral_bounds.phi.bot').get_parameter_value().double_value

        self.params['~d_resolution'] = self.get_parameter('d_resolution')\
            .get_parameter_value().double_value
        self.params['~phi_resolution'] = self.get_parameter('phi_resolution')\
            .get_parameter_value().double_value

        self.params['~stop_line_slowdown']['start'] = self.get_parameter(
            'stop_line_slowdown.start').get_parameter_value().double_value
        self.params['~stop_line_slowdown']['end'] = self.get_parameter(
            'stop_line_slowdown.end').get_parameter_value().double_value

    def publish_cmd(self, msg: Twist2DStamped):
        """Publish a message on the `car_cmd` topic.

        :param msg: Message to publish.
        :type msg: Twist2DStamped
        """
        self.pub_car_cmd.publish(msg)

    def param_update_cb(self, params) -> SetParametersResult:
        """Parameter update callback

        :param params: List of parameters
        :type params: _type_
        :return: If param update was successful.
        :rtype: SetParametersResult
        """
        success = False
        for param in params:
            if param.name in LaneControllerNode.ROS_PARAM_NAMES:
                name_parts = param.name.split('.')
                name_parts[0] = f'~{name_parts[0]}'

                ptr = self.params
                # Get the nested dictionary
                for i in range(len(name_parts)-1):
                    ptr = ptr[name_parts[i]]
                # Access the nested dictionary and update the value
                ptr[name_parts[-1]] = param.value
                success = True

        if success:
            self.controller.update_parameters(self.params)

        return SetParametersResult(successful=success)

    def all_poses_cb(self, input_pose_msg: LanePose, pose_source: str):
        """Callback receiving pose messages from multiple topics.
        If the source of the message corresponds with the current wanted pose
        source, it computes a control command.

        :param input_pose_msg: Message containing information about the current
            lane pose.
        :type msg: LanePose
        :param pose_source: Source of the message, specified in the subscriber.
        :type type: str
        """
        if pose_source == self.current_pose_source:
            self.pose_msg_dict[pose_source] = input_pose_msg
            self.pose_msg = input_pose_msg
            self.get_control_action(self.pose_msg)

    def wheels_cmd_executed_cb(self, msg: WheelsCmdStamped):
        """Callback that reports if the requested control action was executed

        :param msg: Executed wheel commands
        :type msg: WheelsCmdStamped
        """
        if not self.received_wheels_cmd_executed:
            self.received_wheels_cmd_executed = True
            self.get_logger().info("Received first wheels_cmd callback")
        self.wheels_cmd_executed = msg

    def stop_line_reading_cb(self, msg: StopLineReading):
        """Callback storing current distance to the next stopline,
        if one is detected.

        :param msg: Message containing information about the next stop line.
        :type msg: StopLineReading
        """
        self.stop_line_distance = np.sqrt(msg.stop_line_point.x ** 2 +
            msg.stop_line_point.y ** 2)
        self.stop_line_detected = msg.stop_line_detected
        self.at_obstacle_stop_line = msg.at_stop_line

    def obstacle_stop_line_reading_cb(self, msg: StopLineReading):
        """Callback storing the current obstacle distance, if detected.

        :param msg: Message containing information about the virtual
            obstacle stopline.
        :type msg: StopLineReading
        """
        self.obstacle_stop_line_distance = np.sqrt(msg.stop_line_point.x ** 2
            + msg.stop_line_point.y ** 2)
        self.obstacle_stop_line_detected = msg.stop_line_detected
        self.at_stop_line = msg.at_stop_line

    def get_control_action(self, pose_msg: LanePose):
        """Receives a pose message and updates the related control command.
        Using a controller object, computes the control action using the current
        pose estimate.

        :param pose_msg: Message containing information about the current
            lane pose.
        :type pose_msg: LanePose
        """
        current_s: Time = self.get_clock().now()
        dt: float = None

        if self.last_s is not None:
            # Get time elapsed in seconds
            dt = (current_s - self.last_s).nanoseconds / 1e9

        if self.at_stop_line or self.at_obstacle_stop_line:
            v = 0
            omega = 0
        else:
            # Compute errors
            d_err = pose_msg.d - self.params['~d_offset']
            phi_err = pose_msg.phi

            # We cap the error if it grows too large
            if np.abs(d_err) > self.params['~d_thres']:
                self.get_logger().error("d_err too large, thresholding it!")
                d_err = np.sign(d_err) * self.params['~d_thres']

            wheels_cmd_exec = [self.wheels_cmd_executed.vel_left,
                self.wheels_cmd_executed.vel_right]

            if self.obstacle_stop_line_detected:
                v, omega = self.controller.compute_control_action(
                    d_err, phi_err, dt, wheels_cmd_exec,
                    self.obstacle_stop_line_distance
                )
                # TODO: This is a temporarily fix to avoid vehicle image
                # detection latency caused unable to stop in time.
                v = v * 0.25
                omega = omega * 0.25
            else:
                v, omega = self.controller.compute_control_action(
                    d_err, phi_err, dt, wheels_cmd_exec,
                    self.stop_line_distance
                )

            # For feedforward action (i.e. during intersection navigation)
            omega += self.params["~omega_ff"]

        # Initialize car control msg, add header from input message
        car_control_msg = Twist2DStamped()
        car_control_msg.header = pose_msg.header
        # Add commands to car message
        car_control_msg.v = v
        car_control_msg.omega = omega

        self.publish_cmd(car_control_msg)
        self.last_s = current_s


def main(args=None):
    rclpy.init(args=args)
    node = LaneControllerNode("lane_controller_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
