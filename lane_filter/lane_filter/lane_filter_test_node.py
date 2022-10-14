#!/usr/bin/env python3

import rclpy

from rclpy.node import Node

from dt_interfaces_cps.msg import Segment, SegmentList


class LaneFilterTesterNode(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        pub_fake_segment_list = self.create_publisher(
            SegmentList, "~/segment_list", 1)

        self.get_launch_parameters()

        seg = Segment()
        seg.points[0].x = self._x1
        seg.points[0].y = self._y1
        seg.points[1].x = self._x2
        seg.points[1].y = self._y2

        if self._color == 'white':
            seg.color = Segment.WHITE
        elif self._color == 'yellow':
            seg.color = Segment.YELLOW
        elif self._color == 'red':
            seg.color = Segment.RED
        else:
            self.get_logger().warn("No valid color specified")

        seg_list = SegmentList()
        seg_list.segments.append(seg)
        pub_fake_segment_list.publish(seg_list)

    def get_launch_parameters(self) -> None:
        self.declare_parameter("x1")
        self.declare_parameter("y1")
        self.declare_parameter("x2")
        self.declare_parameter("y2")
        self.declare_parameter("color")

        self._x1 = self.get_parameter("x1").get_parameter_value().double_value
        self._y1 = self.get_parameter("y1").get_parameter_value().double_value
        self._x2 = self.get_parameter("x2").get_parameter_value().double_value
        self._y2 = self.get_parameter("y2").get_parameter_value().double_value
        self._color = self.get_parameter("color").get_parameter_value()\
            .string_value

    def onShutdown(self):
        self.get_logger().info("Shutdown.")


def main(args=None):
    rclpy.init(args=args)
    node = LaneFilterTesterNode("lane_filter_test_node")
    try:
        rclpy.spin_once(node, timeout_sec=5.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.onShutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
