#!/usr/bin/env python3

import rclpy

from rclpy.node import Node
from cv_bridge import CvBridge
from multiprocessing import Lock
from sensor_msgs.msg import CompressedImage

from dt_image_processing_utils import AntiInstagram
from dt_interfaces_cps.msg import AntiInstagramThresholds


class AntiInstagramNode(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.get_launch_params()
    
        # Create publishers
        self.pub = self.create_publisher(
            AntiInstagramThresholds,
            "~/thresholds",
            1)
        
        # Create subscribers
        self.uncorrected_image_subscriber = self.create_subscription(
            CompressedImage,
            "~/uncorrected_image/compressed",
            self.image_cb,
            1)

        self.ai = AntiInstagram()
        self.bridge = CvBridge()
        self.image_msg = None
        self.mutex = Lock()

        # Intialize timer
        _ = self.create_timer(self._interval, self.calculate_new_parameters)

        self.get_logger().info(f"Node initialized with parameters - \
interval: {self._interval}, \
color_balance_scale: {self._color_balance_scale}, \
output_scale: {self._output_scale}")

    def get_launch_params(self):
        """Retrieve the launch parameters."""
        self.declare_parameter("interval")
        self.declare_parameter("color_balance_scale")
        self.declare_parameter("output_scale")

        self._interval = self.get_parameter('interval')\
            .get_parameter_value().double_value
        self._color_balance_scale = self.get_parameter('color_balance_scale')\
            .get_parameter_value().double_value
        self._output_scale = self.get_parameter('output_scale')\
            .get_parameter_value().double_value


    def image_cb(self, msg: CompressedImage):
        """Callback function for image subscriber.

        :param msg: Message containing the compressed image.
        :type msg: CompressedImage
        """
        with self.mutex:
            self.image_msg = msg
    
    def decode_image_msg(self):
        """Use CvBridge to uncompress the original compressed image.

        :return: Uncompressed image
        :rtype: _type_
        """
        with self.mutex:
            try:
                return self.bridge.compressed_imgmsg_to_cv2(
                    self.image_msg, 'bgr8')
            except ValueError as e:
                self.get_logger().error(
                    f"Anti_instagram cannot decode image: {e}")
                return None

    def calculate_new_parameters(self):
        """Calculate the color balance thresholds and publish the results
        on the ROS topic.
        """
        if self.image_msg is None:
            self.get_logger().debug(f"Waiting for first image!")
            return
        image = self.decode_image_msg()
        if image is None:
            return

        lower_thresholds, higher_thresholds = \
            self.ai.calculate_color_balance_thresholds(
                image, self._output_scale, self._color_balance_scale)
        # convert values from numpy.uint8 to int
        lower_thresholds = [int(val) for val in lower_thresholds]

        # Publish parameters
        msg = AntiInstagramThresholds()
        msg.low = lower_thresholds
        msg.high = higher_thresholds
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AntiInstagramNode("anti_instagram_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
