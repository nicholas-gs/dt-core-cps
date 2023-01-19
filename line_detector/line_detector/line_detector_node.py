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

from dt_image_processing_utils import AntiInstagram
from dt_image_processing_utils import normalize_lines
from line_detector import (
    plotMaps,
    plotSegments,
    ColorRange,
    LineDetector
)
from dt_ood_interfaces_cps.msg import (
    BoundedLine,
    DetectorInput
)
from dt_interfaces_cps.msg import (
    Segment,
    Vector2D,
    SegmentList,
    AntiInstagramThresholds
)


class LineDetectorNode(Node):
    """
    The ``LineDetectorNode`` is responsible for detecting the line white, yellow
    and red line segment in an image and is used for lane localization.
    Upon receiving an image, this node reduces its resolution, cuts off the top
    part so that only the road-containing part of the image is left, extracts
    the white, red, and yellow segments and publishes them.
    The main functionality of this node is implemented in the
    :py:class:`line_detector.LineDetector` class.
    The performance of this node can be very sensitive to its configuration
    parameters. So you can tune the values in the configuration YAML file.
    Debug publishers are also present to make debugging easier. To reduce
    bandwidth, messages are only published on these topics if there are at
    least 1 subscriber.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that
            ROS will use
    Configuration:
        ~line_detector_parameters: A dictionary with the
            parameters for the detector.
        ~colors: A dictionary of colors and color ranges to be detected in the
            image. The keys (color names) should match the ones in the Segment
            message definition, otherwise an exception will be thrown!See the
            ``config`` directory in the node code for the default ranges.
        ~img_size: The desired downsized resolution of the image. Lower
            resolution would result in faster detection but lower performance,
            default is ``[120,160]``
        ~top_cutoff: The number of rows to be removed from the top of the image
            after resizing, default is 40
    Subscriber:
        ~camera_node/image/compressed: The camera images
        ~anti_instagram_node/thresholds: The thresholds to do color correction
    Publishers:
        ~segment_list: A list of the detected segments.
        ~ood_input: Input for the Out-of-Distribution detector
        ~debug/segments: Debug topic with the segments drawn on the
            input image
        ~debug/edges: Debug topic with the Canny edges drawn on the
            input image
        ~debug/maps: Debug topic with the regions falling in each
            color range drawn on the input image
        ~debug/ranges_HS: Debug topic with a histogram of the colors in the
            input image and the color ranges, Hue-Saturation projection
        ~debug/ranges_SV: Debug topic with a histogram of the colors in the
            input image and the color ranges, Saturation-Value projection
        ~debug/ranges_HV: Debug topic with a histogram of the colors in the
            input image and the color ranges, Hue-Value projection
    """
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.get_launch_configs()
        self.load_config_file(self._param_file_path)

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
        self.color_ranges = {color : ColorRange.fromDict(d,
            self._color_representations[color]) for color, d in
            list(self._colors.items())}

        # Publishers
        self.pub_lines = self.create_publisher(
            SegmentList,
            "~/segment_list",
            1)
        self.pub_ood = self.create_publisher(
            DetectorInput,
            "~/ood_input",
            1)
        self.pub_d_hough = self.create_publisher(
            Image,
            "~/debug/hough",
            1)
        self.pub_d_segments = self.create_publisher(
            Image,
            "~/debug/segments",
            1)
        self.pub_d_edges = self.create_publisher(
            Image,
            "~/debug/edges",
            1)
        self.pub_d_maps = self.create_publisher(
            Image,
            "~/debug/maps",
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

    def get_launch_configs(self) -> None:
        self.declare_parameter("param_file_path")
        self.declare_parameter("output_to_ood")

        self._param_file_path = self.get_parameter("param_file_path")\
            .get_parameter_value().string_value
        self._output_to_ood = self.get_parameter("output_to_ood")\
            .get_parameter_value().string_value

        valid_outputs = ["raw", "canny"]
        if self._output_to_ood not in valid_outputs:
            self.get_logger().warn(f"{self._output_to_ood} \
not a valid configuration in {valid_outputs}, using 'raw' as default")
            self._output_to_ood = "raw"

    def load_config_file(self, file_path: str):
        with open(file=file_path) as f:
            data = yaml.safe_load(f)
        self._line_detector_parameters = data.get('line_detector_parameters')
        self._colors = data.get('colors')
        self._color_representations = data.get('color_representations')
        self._top_cutoff = data.get('top_cutoff')
        self._img_size = data.get('img_size')

        self.get_logger().debug(f"Initialized with parameters: \
_output_to_ood: {self._output_to_ood}, \
_line_detector_parameters: {self._line_detector_parameters}, \
_colors: {self._colors}, \
_color_representations: {self._color_representations}, \
_top_cutoff: {self._top_cutoff}, \
_img_size: {self._img_size}")

    def thresholds_cb(self, threshold_msg: AntiInstagramThresholds):
        self.anti_instagram_thresholds['lower'] = threshold_msg.low
        self.anti_instagram_thresholds['higher'] = threshold_msg.high
        self.ai_thresholds_received = True

    def image_cb(self, image_msg: CompressedImage):
        """Processes the incoming image messages.
        Performs the following steps for each incoming image:
        #. Performs color correction
        #. Resizes the image to the ``~img_size`` resolution
        #. Removes the top ``~top_cutoff`` rows in order to remove the part of the image that doesn't include the road
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

        if self.pub_lines.get_subscription_count() > 0:
            # Construct a SegmentList
            segment_list = SegmentList()
            segment_list.header.stamp = image_msg.header.stamp

            # Fill in the segment_list with all the detected segments
            for color, det in list(detections.items()):
                # Get the ID for the color from the Segment msg definition
                # Throw and exception otherwise
                if len(det.lines) > 0 and len(det.normals) > 0:
                    try:
                        color_id = getattr(Segment, color)
                        lines_normalized = normalize_lines(
                            det.lines, self._top_cutoff, self._img_size)
                        segment_list.segments.extend(
                            self._to_segment_msg(
                                lines_normalized, det.normals, color_id)
                        )
                    except AttributeError:
                        self.get_logger().error(
                            f"Color name {color} is not defined in the"\
                                " Segment message")

            self.pub_lines.publish(segment_list)

        if self.pub_ood.get_subscription_count() > 0:
            ood_msg = DetectorInput()
            ood_msg.header.stamp = image_msg.header.stamp
            ood_msg.cutoff = self._top_cutoff
            if self._output_to_ood == "raw":
                ood_msg.type = "raw"
                ood_msg.frame = self.bridge.cv2_to_compressed_imgmsg(image)
            elif self._output_to_ood == "canny":
                ood_msg.type = "canny"
                ood_msg.frame = self.bridge.cv2_to_compressed_imgmsg(
                    self.detector.canny_edges)

            bounded_lines = []
            for color, det in list(detections.items()):
                for line in det.lines:
                    # We use the 'unnormalized' lines because the `image`
                    # is resized down and the top cutoff
                    p1 = Vector2D(x=float(line[0]), y=float(line[1]))
                    p2 = Vector2D(x=float(line[2]), y=float(line[3]))
                    bounded_line = BoundedLine(
                        coordinates=[p1,p2], color=getattr(Segment, color))
                    bounded_lines.append(bounded_line)

            ood_msg.lines = bounded_lines
            self.pub_ood.publish(ood_msg)

        self._publish_debug_topics(image, image_msg.header, detections)

    def _publish_debug_topics(self, image, header, detections):
        """If there are any subscribers to the debug topics, generate a debug
        image and publish it.
        """
        if self.pub_d_hough.get_subscription_count() > 0:
            colorrange_detections = {self.color_ranges[c] : det for c, det in
                detections.items()}
            black_frame = np.zeros(image.shape, dtype=np.uint8)
            debug_img = plotSegments(black_frame, colorrange_detections, True)
            debug_image_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            debug_image_msg.header = header
            self.pub_d_hough.publish(debug_image_msg)

        if self.pub_d_segments.get_subscription_count() > 0:
            colorrange_detections = {self.color_ranges[c] : det for c, det in
                detections.items()}
            debug_img = plotSegments(image, colorrange_detections)
            debug_image_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            debug_image_msg.header = header
            self.pub_d_segments.publish(debug_image_msg)

        if self.pub_d_edges.get_subscription_count() > 0:
            debug_image_msg = self.bridge.cv2_to_imgmsg(
                self.detector.canny_edges)
            debug_image_msg.header = header
            self.pub_d_edges.publish(debug_image_msg)

        if self.pub_d_maps.get_subscription_count() > 0:
            black_frame = np.zeros(image.shape, dtype=np.uint8)
            colorrange_detections = {
                self.color_ranges[c]: det for c, det in detections.items()}
            debug_img = plotMaps(black_frame, colorrange_detections)
            debug_image_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            debug_image_msg.header = header
            self.pub_d_maps.publish(debug_image_msg)

        for channels in ["HS", "SV", "HV"]:
            publisher = getattr(self, f"pub_d_ranges_{channels}")
            if publisher.get_subscription_count() > 0:
                debug_img = self._plot_ranges_histogram(channels)
                debug_image_msg = self.bridge.cv2_to_imgmsg(
                    debug_img, encoding="bgr8")
                debug_image_msg.header = header
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
            # Convert from numpy.ndarray to tuple
            color = tuple((int(val) for val in color))
            for i in range(len(color_range.low)):
                pt1 = (
                    int((color_range.high[i, channel_idx[1]] / 2).astype(int)),
                    int((color_range.high[i, channel_idx[0]] / 2).astype(int))
                )
                pt2 = (
                    int((color_range.low[i, channel_idx[1]] / 2).astype(int)),
                    int((color_range.low[i, channel_idx[0]] / 2).astype(int))
                )
                cv2.rectangle(
                    im,
                    pt1=pt1,
                    pt2=pt2,
                    color=color,
                    lineType=cv2.LINE_4,
                )
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
