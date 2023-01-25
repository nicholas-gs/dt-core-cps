#!/usr/bin/env python3

import yaml
import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.time import Time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from dt_interfaces_cps.msg import (
    LanePose,
    SegmentList,
    Twist2DStamped
)
from lane_filter.lane_filter import LaneFilterHistogram


class LaneFilterNode(Node):
    """Generates an estimate of the lane pose.
    Creates a `lane_filter` to get estimates on `d` and `phi`, the lateral and
    heading deviation from the center of the lane. It gets the segments
    extracted by the line_detector as input and output the lane pose estimate.
    """
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.load_config_file(self.get_config_filepath())

        self.filter = LaneFilterHistogram(**self._filter)
        self.bridge = CvBridge()

        self.t_last_update: Time = self.get_clock().now()
        assert(isinstance(self.t_last_update, Time))
        self.current_velocity = None
        self.current_velocity_warn_count = 0
        self.latencyArray = []

        # Subscribers
        self.sub = self.create_subscription(SegmentList, "~/segment_list",
            self.cb_process_segments, 1)
        self.sub_velocity = self.create_subscription(Twist2DStamped,
            "~/car_cmd", self.update_velocity_cb, 1)

        # Publishers
        self.pub_lane_pose = self.create_publisher(LanePose, "~/lane_pose",
            1)
        self.pub_belief_img = self.create_publisher(Image, "~/debug/belief_img",
            1)
        self.pub_seglist_filtered = self.create_publisher(SegmentList,
            "~/debug/seglist_filtered", 1)

        self.get_logger().info("Initialised")

    def get_config_filepath(self) -> str:
        self.declare_parameter("config_file_path")
        return self.get_parameter("config_file_path")\
            .get_parameter_value().string_value

    def load_config_file(self, file_path: str) -> None:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        self._filter = data['lane_filter_histogram_configuration']
        self.get_logger().debug(f"Loaded config file: {self._filter}")

    def cb_process_segments(self, segment_list_msg: SegmentList):
        # Get actual timestamp for latency measurement
        timestamp_before_processing = self.get_clock().now()

        # Step 1: predict
        current_time = self.get_clock().now()
        if self.current_velocity:
            # Get time elapsed in seconds
            dt = ((current_time - self.t_last_update).nanoseconds) / 1e9
            self.filter.predict(dt=dt, v=self.current_velocity.v,
                w=self.current_velocity.omega)
        else:
            # Prevent flooding warning messages
            if self.current_velocity_warn_count % 20 == 0:
                self.get_logger().warn(f"Unable to call `predict` on filter \
because `current_velocity` is None (20/20)")
            self.current_velocity_warn_count += 1

        self.t_last_update = current_time

        # Step 2: update
        self.filter.update(segment_list_msg.segments)

        # Step 3: build messages and publish things
        # TODO: This is getting the key and not the value???
        [d_max, phi_max] = self.filter.getEstimate()

        # Getting the highest belief value from the belief matrix
        max_val = self.filter.getMax()
        # Comparing it to a minimum belief threshold to make sure we are
        # certain enough of our estimate
        # Convert from numpy.bool_ to bool
        in_lane = bool(max_val > self.filter.min_max)

        # build lane pose message to send
        lanePose = LanePose()
        lanePose.header.stamp = segment_list_msg.header.stamp
        lanePose.d = d_max
        lanePose.phi = phi_max
        lanePose.in_lane = in_lane
        lanePose.status = LanePose.NORMAL

        self.pub_lane_pose.publish(lanePose)
        self.debug_output(segment_list_msg,d_max, phi_max,
            timestamp_before_processing)

    def debug_output(self, segment_list_msg: SegmentList, d_max, phi_max,
        timestamp_before_processing: Time) -> None:
        """Creates and publishers debug messages on ROS topics. Will check if
        there are any subscribers before processing the data and publishing it.

        :param segment_list_msg: message containing list of filtered segments
        :type segment_list_msg: SegmentList
        :param d_max: best estimate for d
        :type d_max: _type_
        :param phi_max: best estimate for phi
        :type phi_max: _type_
        :param timestamp_before_processing: timestamp dating from before
            the processing
        :type timestamp_before_processing: _type_
        """
        if self.pub_seglist_filtered.get_subscription_count() > 0:
            # Latency of Estimation including curvature estimation
            estimation_latency_stamp = self.get_clock().now() \
                - timestamp_before_processing
            estimation_latency = estimation_latency_stamp.nanoseconds / 1e9
            self.latencyArray.append(estimation_latency)

            if len(self.latencyArray) >= 20:
                self.latencyArray.pop(0)

            self.get_logger().debug(f"Mean latency of Estimation:............."\
                f"{np.mean(self.latencyArray)}")

            # Get the segments that agree with the best estimate and publish them
            inlier_segments = self.filter.get_inlier_segments(
                segment_list_msg.segments, d_max, phi_max)
            inlier_segments_msg = SegmentList()
            inlier_segments_msg.header = segment_list_msg.header
            inlier_segments_msg.segments = inlier_segments
            self.pub_seglist_filtered.publish(inlier_segments_msg)

        if self.pub_belief_img.get_subscription_count() > 0:
            belief_img = self.bridge.cv2_to_imgmsg(
                np.array(255 * self.filter.belief).astype("uint8"), "mono8")
            belief_img.header.stamp = segment_list_msg.header.stamp
            self.pub_belief_img.publish(belief_img)

    def update_velocity_cb(self, msg: Twist2DStamped):
        """Callback method for current car command message
        """
        if self.current_velocity is None:
            self.get_logger().info("Received first car_cmd callback")
        self.current_velocity = msg


def main(args=None):
    rclpy.init(args=args)
    node = LaneFilterNode("lane_filter_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
