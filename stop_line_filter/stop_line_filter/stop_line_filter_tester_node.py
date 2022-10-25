#!/usr/bin/env python3

import rclpy

from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from dt_interfaces_cps.msg import (
    Segment,
    SegmentList,
)


class LaneFilterTesterNode(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)

        pub_fake_segment_list = self.create_publisher(
            SegmentList,
            "~/segment_list",
            1)

        self.declare_parameters(
            namespace='',
            parameters=[
                ('x1', None, ParameterDescriptor(name='x1',
                    type=ParameterType.PARAMETER_DOUBLE)),
                ('y1', None, ParameterDescriptor(name='y1',
                    type=ParameterType.PARAMETER_DOUBLE)),
                ('x2', None, ParameterDescriptor(name='x2',
                    type=ParameterType.PARAMETER_DOUBLE)),
                ('y2', None, ParameterDescriptor(name='y2',
                    type=ParameterType.PARAMETER_DOUBLE)),
                ('color', 'white', ParameterDescriptor(name='color',
                    type=ParameterType.PARAMETER_STRING)),
            ])

        self.x1 = self.get_parameter('x1').get_parameter_value().double_value
        self.y1 = self.get_parameter('y1').get_parameter_value().double_value
        self.x2 = self.get_parameter('x2').get_parameter_value().double_value
        self.y2 = self.get_parameter('y2').get_parameter_value().double_value
        self.color = self.get_parameter('color')\
            .get_parameter_value().string_value

        seg = Segment()
        seg.points[0].x = self.x1
        seg.points[0].y = self.y1
        seg.points[1].x = self.x2
        seg.points[1].y = self.y2

        self.get_logger().info("Initialized.")

        if self.color == 'while':
            seg.color = Segment.WHITE
        elif self.color == 'yellow':
            seg.color = Segment.YELLOW
        elif self.color == 'red':
            seg.color = Segment.RED
        else:
            self.get_logger().warn('Unknown color specified, going with white')
            seg.color = Segment.WHITE

        seg_list = SegmentList()
        seg_list.segments.append(seg)
        pub_fake_segment_list.publish(seg_list)

        self.get_logger().info("Published message.")

    def on_shutdown(self):
        self.get_logger().info("Shutting down.")


def main(args=None):
    rclpy.init(args=args)
    node = LaneFilterTesterNode("stop_line_filter_test_node")
    try:
        rclpy.spin_once(node, timeout_sec=5.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
