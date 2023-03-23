#!/usr/bin/env python3

import re
import os
import cv2
import sys
import rclpy
import pprint
import numpy as np

import torch
import torchvision

from typing import List
from rclpy.node import Node
from ood_detector import vae
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from dt_ood_utils.device import get_ood_dir

from dt_image_processing_utils import normalize_lines
from dt_ood_cps.cropper import (
    BinCropper,
    TrimCropper
)
from dt_ood_interfaces_cps.msg import (
    OODAlert,
    BoundedLine,
    DetectorInput
)
from dt_interfaces_cps.msg import (
    Segment,
    SegmentList
)

# This line is necessary for unpickle/loading the trained pytorch models.
# https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
sys.modules['vae'] = vae


pp = pprint.PrettyPrinter(indent=2)


CROPPERS = {
    'bin' : BinCropper,
    'trim' : TrimCropper
}

class OoDDetector(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.ood_model = None
        self.bridge = CvBridge()
        self._ood_alert_count = 0

        self.load_launch_parameter()
        self.cropper = CROPPERS[self._crop_type](self._crop_thickness,
            bins=self._dimensions)
        self.load_ood_model(self._ood_model_name)

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
        self.pub_ood_alert = self.create_publisher(
            OODAlert,
            "~/ood_alert",
            1)
        self.pub_cropped_image = self.create_publisher(
            Image,
            "~/debug/cropped_image",
            1)
        self.pub_id_image = self.create_publisher(
            Image,
            "~/debug/id_image",
            1)
        self.pub_ood_image = self.create_publisher(
            Image,
            "~/debug/ood_image",
            1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {device}')

        #Additional Info when using cuda
        if device.type == 'cuda':
            self.get_logger().info(torch.cuda.get_device_name(0))
            self.get_logger().info('Memory Usage:')
            self.get_logger().info(f'Allocated: \
{round(torch.cuda.memory_allocated(0)/1024**3,1)} GB')
            self.get_logger().info(f'Cached: \
{round(torch.cuda.memory_reserved(0)/1024**3,1)} GB')

        self.get_logger().info("Initialized")

    def load_launch_parameter(self):
        self.declare_parameter("ood_model_name")
        self.declare_parameter("ood_threshold")
        self.declare_parameter("alert_threshold")
        self.declare_parameter("crop_type")
        self.declare_parameter("crop_thickness")
        self.declare_parameter("dimensions")

        self._ood_model_name = self.get_parameter("ood_model_name")\
            .get_parameter_value().string_value
        self._ood_threshold = self.get_parameter("ood_threshold")\
            .get_parameter_value().double_value
        self._alert_threshold = self.get_parameter("alert_threshold")\
            .get_parameter_value().double_value
        self._crop_type = self.get_parameter("crop_type")\
            .get_parameter_value().string_value
        self._crop_thickness = self.get_parameter("crop_thickness")\
            .get_parameter_value().integer_value
        _dimens = self.get_parameter("dimensions")\
            .get_parameter_value().integer_array_value
        self._dimensions = [(val, val) for val in _dimens]

        log_str = pp.pformat({
            "ood_model_name" : self._ood_model_name,
            "ood_threshold" : self._ood_threshold,
            "alert_threshold" : self._alert_threshold,
            "crop_type" : self._crop_type,
            "crop_thickness" : self._crop_thickness,
            "dimensions" : self._dimensions})

        self.get_logger().info(f"Launch parameters: {log_str}")
        
        if self._alert_threshold < 0:
            self.get_logger("Since alert threshold is less than 0, OOD alert \
is disabled")

    def load_ood_model(self, file_name: str):
        """Load the trained pytorch model from the shared directory.

        :param file_name: Name of .pt or .pth file
        :type file_name: str
        :raises ValueError: If cannot find matching file name.
        """
        model_path = os.path.join(get_ood_dir(), file_name)
        extensions = ('pt', 'pth')

        if model_path.find('.') == -1:
            model_paths = [f"{model_path}.{ext}" for ext in extensions]
        else:
            model_paths = [model_path]

        loaded_path = None
        for path in model_paths:
            if os.path.exists(path):
                loaded_path = path

                N = int(re.findall("_n[0-9]+_", os.path.split(path)[-1],
                    re.IGNORECASE)[0].split('_')[1][-1])
                model = vae.Vae(
                    conv_blocks=vae.CNN_ARCH,
                    fc_layers=[
                        {
                            'in_neurons': 64,
                            'out_neurons': 32,
                            'activation': torch.nn.LeakyReLU(),
                        },
                        {
                            'in_neurons': 32,
                            'out_neurons': N * 2,
                            'activation': torch.nn.Identity(),
                        }
                    ],
                    beta=vae.BETA
                )
                model.load_state_dict(torch.load(path))
                model.eval()
                self.ood_model = model
                self.get_logger().info(f"OOD Model {loaded_path} loaded")

        if loaded_path is None:
            raise ValueError(f"Trained Pytorch OOD detector cannot be found at \
{pp.pformat(model_paths)}")

    def cb_segments(self, msg: DetectorInput):
        uncompressed_img = self.bridge.compressed_imgmsg_to_cv2(
            msg.frame, "bgr8")
        lines, color_ids = OoDDetector._extract_lines(msg.lines)

        id_lines = []
        id_color_ids = []
        ood_lines = []
        tensor_transform = torchvision.transforms.ToTensor()
        for line, color_ids in zip(lines, color_ids):
            # Fix the size to 32x32 since thats what the VAE model is trained on
            cropped_img = tensor_transform(cv2.resize(
                self.cropper.crop_segments(uncompressed_img, line), (32,32)))
            mu, logvar = self.ood_model.encoder(cropped_img.unsqueeze(0))[:2]
            kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
            if kl_loss < self._ood_threshold:
                id_lines.append(line)
                id_color_ids.append(color_ids)
            else:
                ood_lines.append(line)

        id_lines = np.array(id_lines)

        self.publish_ood_alert(msg.header, id_lines, ood_lines)

        # Publish the filtered lines
        segment_list = SegmentList()
        segment_list.header.stamp = msg.header.stamp

        img_size_height = uncompressed_img.shape[0] + msg.cutoff
        img_size_width =  uncompressed_img.shape[1]

        if len(id_lines) > 0:
            id_lines_normalized = normalize_lines(
                id_lines, msg.cutoff, (img_size_height, img_size_width))
        else:
            id_lines_normalized = []

        segment_list.segments = OoDDetector._to_segment_msg(
            id_lines_normalized, id_color_ids)

        self.pub_segments.publish(segment_list)

        # Debug messages
        if self.pub_cropped_image.get_subscription_count() > 0:
            self.publish_debug_image(
                uncompressed_img, lines, msg.type, self.pub_cropped_image)

        if self.pub_id_image.get_subscription_count() > 0:
            self.publish_debug_image(
                uncompressed_img, id_lines, msg.type, self.pub_id_image)

        if self.pub_ood_image.get_subscription_count() > 0:
            self.publish_debug_image(
                uncompressed_img, ood_lines, msg.type, self.pub_ood_image)

    def publish_debug_image(self, image, lines, image_type, publisher):
        """Publish debug image where image is cropped using the lines."""
        cropped_img = OoDDetector._crop_segments(
            image, lines, self._crop_thickness)
        if (image_type == "canny") and (len(cropped_img.shape) == 2):
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
        publisher.publish(
            self.bridge.cv2_to_imgmsg(cropped_img, encoding="bgr8"))
    
    def publish_ood_alert(self, header, id_lines, ood_lines):
        if self._alert_threshold < 0:
            return
        p = len(ood_lines) / (len(ood_lines) + len(id_lines))
        alert = (p >= self._alert_threshold)
        if alert:
            self._ood_alert_count += 1
        else:
            self._ood_alert_count = 0

        if self._ood_alert_count >= 3:
            self.get_logger().warn(f"Publishing OOD Alert")
            msg = OODAlert(header=header, ood=alert)
            self.pub_ood_alert.publish(msg)
        else:
            return

    @staticmethod
    def _extract_lines(lines: List[BoundedLine]):
        xys = []
        color_ids = []
        for line in lines:
            p1 = line.coordinates[0]
            p2 = line.coordinates[1]
            xys.append([p1.x, p1.y, p2.x, p2.y])
            color_ids.append(line.color)
        return np.array(xys, int), color_ids

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
