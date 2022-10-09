#!/usr/bin/env python3

import cv2
import yaml
import rclpy
import numpy as np

from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import (
    Image,
    CompressedImage
)

# from image_processing.anti_instagram import AntiInstagram
from dt_image_processing_utils import AntiInstagram
from line_detector import (
    plotMaps,
    plotSegments,
    ColorRange,
    LineDetector
)
from dt_interfaces_cps.msg import (
    Segment,
    SegmentList,
    AntiInstagramThresholds
)


class LineDetectorNode(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.load_config_file(self.get_config_filepath())

        self.bridge = CvBridge()

        # The thresholds to be used for AntiInstagram color correction
        self.ai_thresholds_received = False
        self.anti_instagram_thresholds = dict()
        self.ai = AntiInstagram()

        # This holds the colormaps for the debug/ranges images after
        # they are computed once
        self.colormaps = dict()

        # Create a new LineDetector object with the parameters from the
        # Parameter Server / config file
        self.detector = LineDetector(**self._line_detector_parameters)
        # Update the color ranges objects
        self.color_ranges = {color : ColorRange.fromDict(d) for color, d in
            list(self._colors.items())}

        # Publishers
        self.pub_lines = self.create_publisher(SegmentList, "~/segment_list", 1)
        self.pub_d_segments = self.create_publisher(
            CompressedImage,
            "~/debug/segments/compressed",
            1)
        self.pub_d_edges = self.create_publisher(
            CompressedImage,
            "~/debug/edges/compressed",
            1)
        self.pub_d_maps = self.create_publisher(
            CompressedImage,
            "~/debug/maps/compressed",
            1)
        # these are not compressed because compression adds undesired blur
        self.pub_d_ranges_HS = self.create_publisher(
            Image,
            "~/debug/ranges_HS",
            1)
        self.pub_d_ranges_SV = self.create_publisher(
            Image,
            "~/debug/ranges_SV",
            1)
        self.pub_d_ranges_HV = self.create_publisher(
            Image,
            "~/debug/ranges_HV",
            1)

        # Subscribers
        self.sub_images = self.create_subscription(
            CompressedImage,
            "~/image/compressed",
            self.image_cb,
            1)
        self.sub_thresholds = self.create_subscription(
            AntiInstagramThresholds,
            "~/thresholds",
            self.thresholds_cb,
            1)

        self.get_logger().info("Initialized.")

    def get_config_filepath(self) -> str:
        self.declare_parameter("param_file_path")
        return self.get_parameter("param_file_path").get_parameter_value()\
            .string_value

    def load_config_file(self, file_path: str):
        with open(file=file_path) as f:
            data = yaml.safe_load(f)
        self._line_detector_parameters = data.get('line_detector_parameters')
        self._colors = data.get('colors')
        self._top_cutoff = data.get('top_cutoff')
        self._img_size = data.get('img_size')

    def thresholds_cb(self, threshold_msg: AntiInstagramThresholds):
        self.anti_instagram_thresholds['lower'] = threshold_msg.low
        self.anti_instagram_thresholds['higher'] = threshold_msg.high
        self.ai_thresholds_received = True

    def image_cb(self, image_msg: CompressedImage):
        """Processes the incoming image messages.
        Performs the following steps for each incoming image:
        #. Performs color correction
        #. Resizes the image to the ``~img_size`` resolution
        #. Removes the top ``~top_cutoff`` rows in order to remove the part ofthe image that doesn't include the road
        #. Extracts the line segments in the image using :py:class:`line_detector.LineDetector`
        #. Converts the coordinates of detected segments to normalized ones
        #. Creates and publishes the resultant :obj:`duckietown_msgs.msg.SegmentList` message
        #. Creates and publishes debug images if there is a subscriber to the respective topics
        Args:
            image_msg (:obj:`sensor_msgs.msg.CompressedImage`): The receive image message
        """
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as error:
            self.get_logger().error(f"Could not decode image: {error}")
            return

        # Perform color correction
        if self.ai_thresholds_received:
            image = self.ai.apply_color_balance(
                self.anti_instagram_thresholds['lower'],
                self.anti_instagram_thresholds['higher'],
                image)

        # Resize the image to the desired dimensions
        height_original, width_original = image.shape[0:2]
        img_size = (self._img_size[1], self._img_size[0])
        if img_size[0] != width_original or img_size[1] != height_original:
            image = cv2.resize(image, img_size, interpolation=cv2.INTER_NEAREST)
        image = image[self._top_cutoff :, :, :,]

        # Extract the line segments for every color
        self.detector.setImage(image)
        detections = {color : self.detector.detectLines(ranges)
            for color, ranges in list(self.color_ranges.items())}

        # Construct a SegmentList
        segment_list = SegmentList()
        segment_list.header.stamp = image_msg.header.stamp

        # Remove the offset in coordinates coming from the
        # removing of the top part
        arr_cutoff = np.array([0, self._top_cutoff, 0, self._top_cutoff])
        arr_ratio = np.array([
            1.0/self._img_size[1],
            1.0/self._img_size[0],
            1.0/self._img_size[1],
            1.0/self._img_size[0],
        ])

        # Fill in the segment_list with all the detected segments
        for color, det in list(detections.items()):
            # Get the ID for the color from the Segment msg definition
            # Throw and exception otherwise
            if len(det.lines) > 0 and len(det.normals) > 0:
                try:
                    color_id = getattr(Segment, color)
                    lines_normalized = (det.lines + arr_cutoff) * arr_ratio
                    segment_list.segments.extend(
                        self._to_segment_msg(
                            lines_normalized, det.normals, color_id)
                    )
                except AttributeError:
                    self.get_logger().error(
                        f"Color name {color} is not defined in the"\
                            " Segment message")

        # Publish the message
        self.pub_lines.publish(segment_list)

        # If there are any subscribers to the debug topics, generate a debug
        # image and publish it
        if self.pub_d_segments.get_subscription_count() > 0:
            colorrange_detections = {self.color_ranges[c] : det for c, det in
                detections.items()}
            debug_img = plotSegments(image, colorrange_detections)
            debug_image_msg = self.bridge.cv2_to_compressed_imgmsg(debug_img)
            debug_image_msg.header = image_msg.header
            self.pub_d_segments.publish(debug_image_msg)

        if self.pub_d_edges.get_subscription_count() > 0:
            debug_image_msg = self.bridge.cv2_to_compressed_imgmsg(
                self.detector.canny_edges)
            debug_image_msg.header = image_msg.header
            self.pub_d_edges.publish(debug_image_msg)

        if self.pub_d_maps.get_subscription_count() > 0:
            colorrange_detections = {
                self.color_ranges[c]: det for c, det in detections.items()}
            debug_img = plotMaps(image, colorrange_detections)
            debug_image_msg = self.bridge.cv2_to_compressed_imgmsg(debug_img)
            debug_image_msg.header = image_msg.header
            self.pub_d_maps.publish(debug_image_msg)

        # for channels in ["HS", "SV", "HV"]:
        for channels in ["HS", "SV", "HV"]:
            publisher = getattr(self, f"pub_d_ranges_{channels}")
            if publisher.get_subscription_count() > 0:
                debug_img = self._plot_ranges_histogram(channels)
                debug_image_msg = self.bridge.cv2_to_imgmsg(
                    debug_img, encoding="bgr8")
                debug_image_msg.header = image_msg.header
                publisher.publish(debug_image_msg)

    @staticmethod
    def _to_segment_msg(lines, normals, color):
        """Converts line detections to a list of Segment messages.
        Converts the resultant line segments and normals from the
        line detection to a list of Segment messages.
        Args:
            lines (:obj:`numpy array`): An ``Nx4`` array where each
                row represents a line.
            normals (:obj:`numpy array`): An ``Nx2`` array where each
                row represents the normal of a line.
            color (:obj:`str`): Color name string, should be one of the
                pre-defined in the Segment message definition.
        Returns:
            :obj:`list` of :obj:`duckietown_msgs.msg.Segment`: List of
                Segment messages
        """
        segment_msg_list = []
        for x1, y1, x2, y2, norm_x, norm_y in np.hstack((lines, normals)):
            segment = Segment()
            segment.color = color
            segment.pixels_normalized[0].x = x1
            segment.pixels_normalized[0].y = y1
            segment.pixels_normalized[1].x = x2
            segment.pixels_normalized[1].y = y2
            segment.normal.x = norm_x
            segment.normal.y = norm_y
            segment_msg_list.append(segment)
        return segment_msg_list

    def _plot_ranges_histogram(self, channels):
        """Utility method for plotting color histograms and color ranges.
        Args:
            channels (:obj:`str`): The desired two channels,
                should be one of ``['HS','SV','HV']``
        Returns:
            :obj:`numpy array`: The resultant plot image
        """
        channel_to_axis = {"H": 0, "S": 1, "V": 2}
        axis_to_range = {0: 180, 1: 256, 2: 256}

        # Get which is the third channel that will not be shown in this plot
        missing_channel = "HSV".replace(channels[0], "").replace(channels[1], "")

        hsv_im = self.detector.hsv
        # Get the pixels as a list
        # (flatten the horizontal and vertical dimensions)
        hsv_im = hsv_im.reshape((-1, 3))

        channel_idx = [channel_to_axis[channels[0]],
            channel_to_axis[channels[1]]]

        # Get only the relevant channels
        x_bins = np.arange(0, axis_to_range[channel_idx[1]] + 1, 2)
        y_bins = np.arange(0, axis_to_range[channel_idx[0]] + 1, 2)
        h, _, _ = np.histogram2d(
            x=hsv_im[:, channel_idx[0]], y=hsv_im[:, channel_idx[1]], bins=[y_bins, x_bins]
        )
        # Log-normalized histogram
        np.log(h, out=h, where=(h != 0))
        h = (255 * h / np.max(h)).astype(np.uint8)

        # Make a color map, for the missing channel, just take the middle of the range
        if channels not in self.colormaps:
            colormap_1, colormap_0 = np.meshgrid(x_bins[:-1], y_bins[:-1])
            colormap_2 = np.ones_like(colormap_0) * (axis_to_range[channel_to_axis[missing_channel]] / 2)

            channel_to_map = {channels[0]: colormap_0, channels[1]: colormap_1, missing_channel: colormap_2}

            self.colormaps[channels] = np.stack(
                [channel_to_map["H"], channel_to_map["S"], channel_to_map["V"]], axis=-1
            ).astype(np.uint8)
            self.colormaps[channels] = cv2.cvtColor(
                self.colormaps[channels], cv2.COLOR_HSV2BGR)

        # resulting histogram image as a blend of the two images
        im = cv2.cvtColor(h[:, :, None], cv2.COLOR_GRAY2BGR)
        im = cv2.addWeighted(im, 0.5, self.colormaps[channels], 1 - 0.5, 0.0)

        # now plot the color ranges on top
        for _, color_range in list(self.color_ranges.items()):
            # convert HSV color to BGR
            c = color_range.representative
            c = np.uint8([[[c[0], c[1], c[2]]]])
            color = cv2.cvtColor(c, cv2.COLOR_HSV2BGR).squeeze().astype(int)
            for i in range(len(color_range.low)):
                cv2.rectangle(
                    im,
                    pt1=(
                        (color_range.high[i, channel_idx[1]] / 2).astype(np.uint8),
                        (color_range.high[i, channel_idx[0]] / 2).astype(np.uint8),
                    ),
                    pt2=(
                        (color_range.low[i, channel_idx[1]] / 2).astype(np.uint8),
                        (color_range.low[i, channel_idx[0]] / 2).astype(np.uint8),
                    ),
                    color=color,
                    lineType=cv2.LINE_4,
                )
        # ---
        return im


def main(args=None):
    rclpy.init(args=args)
    node = LineDetectorNode("line_detector_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()