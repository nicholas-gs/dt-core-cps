#!/usr/bin/env python3

import cv2
import yaml
import rclpy

from typing import List
from rclpy.node import Node
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import CompressedImage

from dt_interfaces_cps.msg import (
    Segment,
    SegmentList
)
from generic_line_detector import (
    CannyParameters,
    HoughParameters,
    RegionOfInterest,
    GenericLineDetector
)


class GenericLineDetectorNode(Node):
    """A line detector that works in grayscale. Based off:
    https://medium.com/@SunEdition/lane-detection-and-turn-prediction-algorithm-for-autonomous-vehicles-6423f77dc841
    and
    https://towardsdatascience.com/deeppicar-part-4-lane-following-via-opencv-737dd9e47c96.
    """
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.load_config_file(self.get_config_filepath())\

        self.bridge = CvBridge()

        self.line_detector = GenericLineDetector(
            canny_param=self._canny_param,
            hough_param=self._hough_param,
            roi=RegionOfInterest())

        # Publishers
        self.pub_canny_image = self.create_publisher(
            CompressedImage,
            "~/debug/canny_image/compressed",
            1)
        self.pub_hough_segments = self.create_publisher(
            SegmentList,
            "~/debug/hough_segments",
            1)
        self.pub_hough_image = self.create_publisher(
            CompressedImage,
            "~/debug/hough_image/compressed",
            1)
        self.pub_lanes = self.create_publisher(
            SegmentList,
            "~/lanes",
            1)
        self.pub_lane_image = self.create_publisher(
            CompressedImage,
            "~/debug/lane_image/compressed",
            1)

        # Subscribers
        self.sub_images = self.create_subscription(
            CompressedImage,
            "~/image/compressed",
            self.image_cb,
            1)

        self.get_logger().info("Initialized")

    def get_config_filepath(self) -> str:
        self.declare_parameter("param_file_path")
        return self.get_parameter("param_file_path").get_parameter_value()\
            .string_value

    def load_config_file(self, file_path: str):
        with open(file=file_path) as f:
            data = yaml.safe_load(f)

        self._resize_width = data["image_size_width"]
        self._resize_height = data["image_size_height"]
        self._canny_param = CannyParameters(
            threshold_lower=data["canny_edge_parameters"]["threshold_lower"],
            threshold_upper=data["canny_edge_parameters"]["threshold_upper"],
            aperture_size=data["canny_edge_parameters"]["aperture_size"])
        self._hough_param = HoughParameters(
            threshold=data["hough_parameters"]["threshold"],
            min_line_length=data["hough_parameters"]["min_line_length"],
            max_line_gap=data["hough_parameters"]["max_line_gap"])

        self.get_logger().info(f"Read config file - \
_resize_width: {self._resize_width}, \
_resize_height: {self._resize_height}, \
_canny_param: {self._canny_param}, \
_hough_param: {self._hough_param}")

    @staticmethod
    def to_segment_list(coordinates) -> List[Segment]:
        segment_list = []
        for line in coordinates:
            for x1, y1, x2, y2 in line:
                segment_list.append(
                    Segment(points=[Point(x=float(x1), y=float(y1)),
                    Point(x=float(x2), y=float(y2))]))
        return segment_list

    @staticmethod
    def resize_image(image, new_height: int, new_width: int):
        """Resize image. If `new_height` and `new_width` are both 0, then
        no resizing is done.
        """
        if new_height == 0 and new_width == 0:
            return image
        height_original, width_original = image.shape[0:2]
        img_size = (new_width, new_height)
        if img_size[0] != width_original or img_size[1] != height_original:
            image = cv2.resize(image, img_size, interpolation=cv2.INTER_NEAREST)

    def image_cb(self, image_msg: CompressedImage):
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as error:
            self.get_logger().error(f"Could not decode image: {error}")
            return

        image = GenericLineDetectorNode.resize_image(
            image, self._resize_height, self._resize_width)

        # Run line detection pipeline
        self.line_detector.frame = image
        self.line_detector.pipeline()

        # Construct a SegmentList message and publish
        lane_msg = SegmentList()
        lane_msg.header.stamp = image_msg.header.stamp
        lane_msg.segments = GenericLineDetectorNode.to_segment_list(
            self.line_detector.lanes)
        self.pub_lanes.publish(lane_msg)

        if self.pub_canny_image.get_subscription_count() > 0:
            debug_canny_image_msg = self.bridge.cv2_to_compressed_imgmsg(
                self.line_detector.frame_canny)
            debug_canny_image_msg.header = image_msg.header
            self.pub_canny_image.publish(debug_canny_image_msg)

        if self.pub_hough_segments.get_subscription_count() > 0:
            debug_hough_segments_msg = SegmentList()
            debug_hough_segments_msg.header.stamp = image_msg.header.stamp
            debug_hough_segments_msg.segments = GenericLineDetectorNode\
                .to_segment_list(self.line_detector.segments)
            self.pub_hough_segments.publish(debug_hough_segments_msg)

        if self.pub_hough_image.get_subscription_count() > 0:
            debug_hough_image_msg = self.bridge.cv2_to_compressed_imgmsg(
                self.line_detector.frame_segment)
            debug_hough_image_msg.header = image_msg.header
            self.pub_hough_image.publish(debug_hough_image_msg)

        if self.pub_lane_image.get_subscription_count() > 0:
            debug_lane_image_msg = self.bridge.cv2_to_compressed_imgmsg(
                self.line_detector.frame_lanes)
            debug_lane_image_msg.header = image_msg.header
            self.pub_lane_image.publish(debug_lane_image_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GenericLineDetectorNode("generic_line_detector_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
