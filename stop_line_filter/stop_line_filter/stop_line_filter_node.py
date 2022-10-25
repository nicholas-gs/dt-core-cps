#!/usr/bin/env python3

import time
import rclpy
import numpy as np

from rclpy.node import Node
from geometry_msgs.msg import Point
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from dt_interfaces_cps.msg import (
    LanePose,
    FSMState,
    Segment,
    SegmentList,
    BoolStamped,
    StopLineReading
)


class StopLineFilterNode(Node):

    INTERSECTION_CONTROL = 'INTERSECTION_CONTROL'
    LANE_FOLLOWING = 'LANE_FOLLOWING'

    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.get_launch_params()

        self.lane_pose = LanePose()
        self.state = 'JOYSTICK_CONTROL'
        self.sleep = False

        # Create publishers
        self.pub_stop_line_reading = self.create_publisher(
            StopLineReading,
            "~/stop_line_reading",
            1)
        self.pub_at_stop_line = self.create_publisher(
            BoolStamped,
            "~/at_stop_line",
            1)

        # Create Subscriptions
        self.sub_segs = self.create_subscription(
            SegmentList,
            "~/segment_list",
            self.cb_segments,
            1)
        self.sub_lane = self.create_subscription(
            LanePose,
            "~/lane_pose",
            self.cb_lane_pose,
            1)
        self.sub_lane = self.create_subscription(
            FSMState,
            "~/fsm_node/mode",
            self.cb_state_change,
            1)

        self.get_logger().info(f"Initialised")

    def get_launch_params(self):
        """Retrieve launch parameters."""
        self.declare_parameters(
            namespace='',
            parameters=[
                ('stop_distance', None, ParameterDescriptor(
                    name='stop_distance', type=ParameterType.PARAMETER_DOUBLE)),
                ('max_y', None, ParameterDescriptor(
                    name='max_y', type=ParameterType.PARAMETER_DOUBLE)),
                ('min_segs', None, ParameterDescriptor(
                    name='min_segs', type=ParameterType.PARAMETER_INTEGER)),
                ('off_time', None, ParameterDescriptor(
                    name='off_time', type=ParameterType.PARAMETER_INTEGER)),
            ])

        self.stop_distance = self.get_parameter('stop_distance')\
            .get_parameter_value().double_value
        self.max_y = self.get_parameter('max_y')\
            .get_parameter_value().double_value
        self.min_segs = self.get_parameter('min_segs')\
            .get_parameter_value().integer_value
        self.off_time = self.get_parameter('off_time')\
            .get_parameter_value().integer_value

    def cb_segments(self, segment_list_msg: SegmentList):
        """Callback for detected line segments.

        :param segment_list_msg: List of line segments detected
        :type segment_list_msg: SegmentList
        """
        if self.sleep:
            return

        good_seg_count = 0
        stop_line_x_accumulator = 0.0
        stop_line_y_accumulator = 0.0

        for segment in segment_list_msg.segments:
            if segment.color != Segment.RED:
                continue
            if segment.points[0].x < 0 or segment.points[1].x < 0:
                # the point is behind us
                continue

            p1_lane = self.to_lane_frame(segment.points[0])
            p2_lane = self.to_lane_frame(segment.points[1])
            avg_x = 0.5 * (p1_lane[0] + p2_lane[0])
            avg_y = 0.5 * (p1_lane[1] + p2_lane[1])
            stop_line_x_accumulator += avg_x
            stop_line_y_accumulator += avg_y
            good_seg_count += 1.0

        stop_line_reading_msg = StopLineReading()
        stop_line_reading_msg.header.stamp = segment_list_msg.header.stamp

        if good_seg_count < self.min_segs:
            stop_line_reading_msg.stop_line_detected = False
            stop_line_reading_msg.at_stop_line = False
            self.pub_stop_line_reading.publish(stop_line_reading_msg)
        else:
            stop_line_reading_msg.stop_line_detected = True
            stop_line_point = Point()
            stop_line_point.x = stop_line_x_accumulator / good_seg_count
            stop_line_point.y = stop_line_y_accumulator / good_seg_count
            stop_line_reading_msg.stop_line_point = stop_line_point
            # Only detect redline if y is within max_y distance:
            stop_line_reading_msg.at_stop_line = (
                stop_line_point.x < self.stop_distance.value and \
                    np.abs(stop_line_point.y) < self.max_y.value)

            self.pub_stop_line_reading.publish(stop_line_reading_msg)
            if stop_line_reading_msg.at_stop_line:
                msg = BoolStamped()
                msg.header.stamp = stop_line_reading_msg.header.stamp
                msg.data = True
                self.pub_at_stop_line.publish(msg)

    def cb_lane_pose(self, msg: LanePose):
        """Callback method for the current lane pose

        :param msg: Message containing the current lane pose
        :type msg: LanePose
        """
        self.lane_pose = msg

    def cb_state_change(self, msg: FSMState):
        """Callback for FSM state change

        :param msg: Message containing the new FSM state
        :type msg: FSMState
        """
        if ((self.state == StopLineFilterNode.INTERSECTION_CONTROL)
            and (msg.state == StopLineFilterNode.LANE_FOLLOWING)):
            self.after_intersection_work()
        self.state = msg.state

    def after_intersection_work(self):
        """Temporarily block stop line detection after crossing an intersection.
        """
        self.get_logger().info("Blocking stop line detection after intersection")
        stop_line_reading_msg = StopLineReading()
        stop_line_reading_msg.stop_line_detected = False
        stop_line_reading_msg.at_stop_line = False
        self.pub_stop_line_reading.publish(stop_line_reading_msg)
        self.sleep = True
        time.sleep(float(self.off_time)) # Probably not the best idea.
        self.sleep = False
        self.get_logger().info("Resuming stop line detection after intersection")

    def to_lane_frame(self, point):
        p_homo = np.array([point.x, point.y, 1])
        phi = self.lane_pose.phi
        d = self.lane_pose.d
        T = np.array([[np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), d], [0, 0, 1]])
        p_new_homo = T.dot(p_homo)
        p_new = p_new_homo[0:2]
        return p_new


def main(args=None):
    rclpy.init(args=args)
    node = StopLineFilterNode("stop_line_filter_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
