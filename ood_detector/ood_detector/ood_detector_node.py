#!/usr/bin/env python3

import cv2
import rclpy
import numpy as np

from typing import List
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from dt_image_processing_utils import normalize_lines
from dt_ood_interfaces_cps.msg import (
    BoundedLine,
    DetectorInput
)
from dt_interfaces_cps.msg import (
    Segment,
    SegmentList
)


class OoDDetector(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.bridge = CvBridge()

        # Subscribers
        self.sub_segments = self.create_subscription(
            DetectorInput,
            "~/ood_input",
            self.cb_segments,
            1)

        # Publishers
        self.pub_segments = self.create_publisher(
            SegmentList,
            "~/ood_segment_list",
            1)
        self.pub_image = self.create_publisher(
            Image,
            "~/debug/crop_image",
            1)

        self.get_logger().info("Initialized")

    def cb_segments(self, msg: DetectorInput):
        uncompressed_img = self.bridge.compressed_imgmsg_to_cv2(
            msg.frame, "bgr8")
        lines, color_ids = OoDDetector._extract_lines(msg.lines)

        filtered_lines = lines

        segment_list = SegmentList()
        segment_list.header.stamp = msg.header.stamp

        img_size_height = uncompressed_img.shape[0] + msg.cutoff
        img_size_width =  uncompressed_img.shape[1]

        filtered_lines_normalized = normalize_lines(
            filtered_lines, msg.cutoff, (img_size_height, img_size_width))
        segment_list.segments = OoDDetector._to_segment_msg(
            filtered_lines_normalized, color_ids)

        self.pub_segments.publish(segment_list)

        if self.pub_image.get_subscription_count() > 0:
            cropped_img = OoDDetector._crop_segments(
                uncompressed_img, filtered_lines, 2)
            if (msg.type == "canny") and (len(cropped_img.shape) == 2):
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
            self.pub_image.publish(
                self.bridge.cv2_to_imgmsg(cropped_img, encoding="bgr8"))

    @staticmethod
    def _extract_lines(lines: List[BoundedLine]):
        xys = []
        color_ids = []
        for line in lines:
            p1 = line.coordinates[0]
            p2 = line.coordinates[1]
            xys.append([p1.x, p1.y, p2.x, p2.y])
            color_ids.append(line.color)
        return np.array(xys, np.int), color_ids

    @staticmethod
    def _to_segment_msg(lines, color_ids):
        """Converts line detections to a list of Segment messages.
        Converts the resultant line segments and normals from the
        line detection to a list of Segment messages.
        Args:
            lines (:obj:`numpy array`): An ``Nx4`` array where each
                row represents a line.
            color_ids (:obj:`list`): Color ids, should be one of the
                pre-defined in the Segment message definition.
        Returns:
            :obj:`list` of :obj:`duckietown_msgs.msg.Segment`: List of
                Segment messages
        """
        segment_msg_list = []
        for idx, line in enumerate(lines):
            x1, y1, x2, y2 = line
            segment = Segment()
            segment.color = color_ids[idx]
            segment.pixels_normalized[0].x = x1
            segment.pixels_normalized[0].y = y1
            segment.pixels_normalized[1].x = x2
            segment.pixels_normalized[1].y = y2
            segment_msg_list.append(segment)
        return segment_msg_list

    @staticmethod
    def _crop_segments(frame, lines: np.array, thickness: int=1):
        """Crop an image using the list of bounded lines.
        """
        background = np.zeros_like(frame)
        for line in lines:
            background = cv2.line(
                background, (line[0], line[1]), (line[2], line[3]),
                color=(255,255,255), thickness=thickness)
        return cv2.bitwise_and(background, frame)

def main(args=None):
    rclpy.init(args=args)
    node = OoDDetector("ood_detector_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
