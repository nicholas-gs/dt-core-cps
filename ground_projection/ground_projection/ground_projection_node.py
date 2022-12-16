#!/usr/bin/env python3

import cv2
import rclpy
import numpy as np

from typing import Optional

from rclpy.node import Node
from cv_bridge import CvBridge
from geometry_msgs.msg import Point as PointMsg
from sensor_msgs.msg import CameraInfo, CompressedImage

from dt_calibration_utils import read_calibration
from dt_interfaces_cps.msg import Segment, SegmentList
from dt_image_processing_utils import (
    Point,
    Rectify,
    GroundProjectionGeometry
)


class GroundProjectionNode(Node):
    """This node projects the line segments detected in the image to the ground
    plane and in the robot's reference frame. In this way it enables lane
    localization in the 2D ground plane. This projection is performed using the
    homography matrix obtained from the extrinsic calibration procedure.
    Subscribers:
        ~camera_info (:obj:`sensor_msgs.msg.CameraInfo`): Intrinsic properties
            of the camera. Needed for rectifying the segments.
        ~lineseglist_in (:obj:`duckietown_msgs.msg.SegmentList`): Line segments
            in pixel space from unrectified images
    Publishers:
        ~lineseglist_out (:obj:`duckietown_msgs.msg.SegmentList`): Line segments
            in the ground plane relative to the robot origin
        ~debug/ground_projection_image/compressed
            (:obj:`sensor_msgs.msg.CompressedImage`): Debug image that shows the
            robot relative to the projected segments. Useful to check if the
            extrinsic calibration is accurate. Only published on if there is at
            least 1 subscriber.
    """
    bridge: CvBridge
    recitier: Optional[Rectify]
    ground_projector: Optional[GroundProjectionGeometry]

    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.get_launch_parameters()

        self.bridge = CvBridge()
        self.ground_project = None
        self.rectifier = None
        self.homograph = self.load_extrinsics()
        self.first_processing_done = False
        self.camera_info_received = False

        # Subscribers
        self.sub_camera_info = self.create_subscription(
            CameraInfo,
            "~/camera_info",
            self.cb_camera_info, 1)
        self.sub_lineseglist = self.create_subscription(
            SegmentList,
            "~/lineseglist_in",
            self.lineseglist_cb,
            1)

        # Publishers
        self.pub_lineseglist = self.create_publisher(
            SegmentList,
            "~/lineseglist_out",
            1)
        self.pub_debug_img = self.create_publisher(
            CompressedImage,
            "~/debug/ground_project_image/compressed",
            1)

        self.bridge = CvBridge()
        self.debug_img_bg = None

        self.get_logger().info("Initialized")

    def get_launch_parameters(self) -> None:
        self.declare_parameter("veh")
        self._veh = self.get_parameter("veh").get_parameter_value().string_value

    def cb_camera_info(self, msg: CameraInfo):
        """Initializes a :py:class:`image_processing.GroundProjectionGeometry`
        object and a :py:class:`image_processing.Rectify` object for image
        rectification.
        Args:
            msg (:obj:`sensor_msgs.msg.CameraInfo`): Intrinsic properties of
                the camera.
        """
        if not self.camera_info_received:
            self.rectifier = Rectify(msg)
            self.ground_projector = GroundProjectionGeometry(
                im_width=msg.width,
                im_height=msg.height,
                homography=np.array(self.homograph).reshape(3,3))
        self.camera_info_received = True

    def pixel_msg_to_ground_msg(self, point_msg) -> PointMsg:
        """Creates a :py:class:`ground_projection.Point` object from a
        normalized point message from an unrectified image. It converts it to
        pixel coordinates and rectifies it. Then projects it to the ground plane
        and converts it to a ROS Point message.
        Args:
            point_msg (:obj:`geometry_msgs.msg.Point`): Normalized point
                coordinates from an unrectified image.
        Returns:
            :obj:`geometry_msgs.msg.Point`: Point coordinates in the ground
                reference frame.
        """
        # normalized coordinates to pixel:
        norm_pt = Point.from_message(point_msg)
        pixel = self.ground_projector.vector2pixel(norm_pt)
        # rectify
        rect = self.rectifier.rectify_point(pixel)
        # convert to Point
        rect_pt = Point.from_message(rect)
        # project on ground
        ground_pt = self.ground_projector.pixel2ground(rect_pt)
        # point to message
        ground_pt_msg = PointMsg()
        ground_pt_msg.x = ground_pt.x
        ground_pt_msg.y = ground_pt.y
        ground_pt_msg.z = ground_pt.z

        return ground_pt_msg

    def lineseglist_cb(self, seglist_msg: SegmentList):
        """Projects a list of line segments on the ground reference frame point
        by point by calling :py:meth:`pixel_msg_to_ground_msg`. Then publishes
        the projected list of segments.
        Args:
            seglist_msg (:obj:`duckietown_msgs.msg.SegmentList`): Line segments
            in pixel space from unrectified images
        """
        if self.camera_info_received:
            seglist_out = SegmentList()
            seglist_out.header = seglist_msg.header
            for received_segment in seglist_msg.segments:
                new_segment = Segment()
                new_segment.points[0] = self.pixel_msg_to_ground_msg(
                    received_segment.pixels_normalized[0])
                new_segment.points[1] = self.pixel_msg_to_ground_msg(
                    received_segment.pixels_normalized[1])
                new_segment.color = received_segment.color
                # TODO what about normal and points
                seglist_out.segments.append(new_segment)
            self.pub_lineseglist.publish(seglist_out)

            if not self.first_processing_done:
                self.get_logger().info("First projected segments published.")
                self.first_processing_done = True

            if self.pub_debug_img.get_subscription_count() > 0:
                debug_image_msg = self.bridge.cv2_to_compressed_imgmsg(
                    self.debug_image(seglist_out))
                debug_image_msg.header = seglist_out.header
                self.pub_debug_img.publish(debug_image_msg)
        else:
            self.get_logger().warn("Not processing callback message,"\
                " Waiting for a CameraInfo message")

    def debug_image(self, seg_list):
        """Generates a debug image with all the projected segments plotted with
        respect to the robot's origin.
        Args:
            seg_list (:obj:`duckietown_msgs.msg.SegmentList`): Line segments in
                the ground plane relative to the robot origin
        Returns:
            :obj:`numpy array`: an OpenCV image
        """
        # dimensions of the image are 1m x 1m so, 1px = 2.5mm
        # the origin is at x=200 and y=300

        # if that's the first call, generate the background
        if self.debug_img_bg is None:
            # initialize gray image
            self.debug_img_bg = np.ones((400, 400, 3), np.uint8) * 128
            # draw vertical lines of the grid
            for vline in np.arange(40, 361, 40):
                cv2.line(
                    self.debug_img_bg, pt1=(vline, 20), pt2=(vline, 300),
                    color=(255, 255, 0), thickness=1)

            # draw the coordinates
            cv2.putText(
                self.debug_img_bg,
                "-20cm",
                (120 - 25, 300 + 15),
                cv2.FONT_HERSHEY_PLAIN,
                0.8,
                (255, 255, 0),
                1,)
            cv2.putText(
                self.debug_img_bg,
                "  0cm",
                (200 - 25, 300 + 15),
                cv2.FONT_HERSHEY_PLAIN,
                0.8,
                (255, 255, 0),
                1,)
            cv2.putText(
                self.debug_img_bg,
                "+20cm",
                (280 - 25, 300 + 15),
                cv2.FONT_HERSHEY_PLAIN,
                0.8,
                (255, 255, 0),
                1,)
            # draw horizontal lines of the grid
            for hline in np.arange(20, 301, 40):
                cv2.line(
                    self.debug_img_bg, pt1=(40, hline), pt2=(360, hline),
                    color=(255, 255, 0), thickness=1)
            # draw the coordinates
            cv2.putText(
                self.debug_img_bg, "20cm", (2, 220 + 3),
                cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)
            cv2.putText(
                self.debug_img_bg, " 0cm", (2, 300 + 3),
                cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)
            # draw robot marker at the center
            cv2.line(
                self.debug_img_bg,
                pt1=(200 + 0, 300 - 20),
                pt2=(200 + 0, 300 + 0),
                color=(255, 0, 0),
                thickness=1,)
            cv2.line(
                self.debug_img_bg,
                pt1=(200 + 20, 300 - 20),
                pt2=(200 + 0, 300 + 0),
                color=(255, 0, 0),
                thickness=1,)
            cv2.line(
                self.debug_img_bg,
                pt1=(200 - 20, 300 - 20),
                pt2=(200 + 0, 300 + 0),
                color=(255, 0, 0),
                thickness=1,)

        # map segment color variables to BGR colors
        color_map = {Segment.WHITE: (255, 255, 255), Segment.RED: (0, 0, 255),
            Segment.YELLOW: (0, 255, 255)}

        image = self.debug_img_bg.copy()

        # plot every segment if both ends are in the scope of the image
        # (within 50cm from the origin)
        for segment in seg_list.segments:
            if not np.any(
                np.abs([segment.points[0].x, segment.points[0].y,
                    segment.points[1].x, segment.points[1].y])
                > 0.50
            ):
                cv2.line(
                    image,
                    pt1=(int(segment.points[0].y * -400) + 200,
                        int(segment.points[0].x * -400) + 300),
                    pt2=(int(segment.points[1].y * -400) + 200,
                        int(segment.points[1].x * -400) + 300),
                    color=color_map.get(segment.color, (0, 0, 0)),
                    thickness=1,
                )

        return image

    def load_extrinsics(self):
        """Loads the homography matrix from the extrinsic calibration file.
        Returns:
            :obj:`numpy array`: the loaded homography matrix
        """
        calibration_data, full_fp = read_calibration(
            f"camera_extrinsic/{self._veh}.yaml")
        if calibration_data is None:
            self.get_logger().warn(f"Cannot import calibration data from"\
            f" {full_fp}, trying default file.")
            calibration_data, full_fp = read_calibration(
                f"camera_extrinsic/default.yaml")
            if calibration_data is None:
                self.get_logger().fatal("Cannot import default calibration"\
                    f" data from {full_fp}")
                raise RuntimeError("No calibration file found")

        return calibration_data["homography"]


def main(args=None):
    rclpy.init(args=args)
    node = GroundProjectionNode("ground_projection_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
