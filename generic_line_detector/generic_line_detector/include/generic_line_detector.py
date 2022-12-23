#!/usr/bin/env python3

import sys
import cv2
import numpy as np

from typing import Tuple
from dataclasses import fields, dataclass


__all__ = [
    "CannyParameters",
    "HoughParameters",
    "RegionOfInterest",
    "GenericLineDetector"
]


@dataclass
class CannyParameters:
    threshold_lower: int
    threshold_upper: int
    aperture_size: int


@dataclass
class HoughParameters:
    threshold: int
    min_line_length: int
    max_line_gap: int


@dataclass
class RegionOfInterest:
    """Represents the 4 corners of a trapezoid. Each point is represented using
    (x, y), where each value is the percentage with range [0.0, 1.0], with
    (0.0, 0.0) representing the bottom left corner of the image, and (1.0, 1.0)
    representing the top right corner of the image.
    """
    bottom_left: Tuple[float, float] = (0.0, 0.0)
    bottom_right: Tuple[float, float] = (1.0, 0.0)
    top_left: Tuple[float, float] = (0.1, 0.5)
    top_right: Tuple[float, float] = (0.9, 0.5)

    def __post_init__(self):
        invalid_range = lambda x : x < 0.0 or x > 1.0
        field_names = [field.name for field in fields(RegionOfInterest)]
        for field in field_names:
            x, y = getattr(self, field)
            if invalid_range(x) or invalid_range(y):
                raise ValueError("Valid range [0.0 <= v <= 1.0]")

    def to_polygon(self, height, width):
        """Convert the 4 points into a `np.array` that can be fed into the
        `cv2.fillPoly` function.
        """
        assert (height <= width)
        return np.array([[
            # Bottom-left point
            (int(width*self.bottom_left[0]), height*(1-self.bottom_left[1])),
            # Top-left point
            (int(width*self.top_left[0]), int(height*(1-self.top_left[1]))),
            # Top-right point
            (int(width*self.top_right[0]), int(height*(1-self.top_right[1]))),
            # Bottom-right point
            (int(width*self.bottom_right[0]), height*(1-self.bottom_right[1])),
        ]], np.int32)


class GenericLineDetector:
    """The ``GenericLineDetector`` is responsible for detecting lanes in an
    image. It processes the image as the grayscale.
    Attirbutes/Results:
        frame_canny: Canny edges on top of the original image.
        segments: List of all segments found by Hough transform.
        frame_segment: All segments found by Hough transform on top of the
            original image.
        lanes: List of all lanes found. Derived by processing the output of
            Hough transform.
        frame_lanes: Lanes on top of the original image.
    """
    def __init__(
        self,
        canny_param: CannyParameters,
        hough_param: HoughParameters,
        roi: RegionOfInterest,
    ) -> None:
        """_summary_

        :param canny_param: _description_
        :type canny_param: CannyParameters
        :param hough_param: _description_
        :type hough_param: HoughParameters
        :param roi: _description_
        :type roi: RegionOfInterest
        """
        self.canny_param = canny_param
        self.hough_param = hough_param
        self.roi = roi
        self._frame = None
        self._reset_results()

    def _reset_results(self):
        """Reset all results."""
        self._frame_canny = None
        self._segments = list()
        self._frame_segment = None
        self._lanes = list()
        self._frame_lanes = None

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, frame):
        self._frame = frame

    @property
    def frame_canny(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """
        return self._frame_canny

    @property
    def segments(self):
        """All line segments found using the Hough algorithm

        :return: _description_
        :rtype: _type_
        """
        return self._segments

    @property
    def frame_segment(self):
        """Image with all line segments found using the Hough algorithm on top.

        :return: _description_
        :rtype: _type_
        """
        return self._frame_segment

    @property
    def lanes(self):
        """Lanes found.

        :return: _description_
        :rtype: _type_
        """
        return self._lanes

    @property
    def frame_lanes(self):
        """Image with all lanes found on top.

        :return: _description_
        :rtype: _type_
        """
        return self._frame_lanes

    @staticmethod
    def apply_blur(frame):
        return cv2.blur(frame, (3,3))

    @staticmethod
    def region_of_interest(frame, polygon):
        """_summary_

        :param frame: _description_
        :type frame: _type_
        :return: _description_
        :rtype: _type_
        """
        height, width = frame.shape

        mask = np.zeros_like(frame)

        # Fill polygon with value 255 (white color)
        cv2.fillPoly(mask, polygon, 255)

        # AND bitwise the mask and the original frame
        return cv2.bitwise_and(frame, mask)

    @staticmethod
    def make_points(frame, line):
        """_summary_

        :param frame: _description_
        :type frame: _type_
        :param line: _description_
        :type line: _type_
        :return: _description_
        :rtype: _type_
        """
        height, width, _ = frame.shape
        slope, intercept = line
        y1 = height  # bottom of the frame
        y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

        # bound the coordinates within the frame
        x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
        x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
        return [[x1, y1, x2, y2]]

    @staticmethod
    def average_slope_intercept(frame, line_segments):
        """Combine line segments into one or two lane lines
        If all line slopes are < 0: then we only have detected left lane
        If all line slopes are > 0: then we only have detected right lane

        :param frame: _description_
        :type frame: _type_
        :param line_segments: _description_
        :type line_segments: _type_
        :return: _description_
        :rtype: _type_
        """
        lane_lines = []
        if line_segments is None:
            return lane_lines

        height, width, _ = frame.shape
        left_fit = []
        right_fit = []

        boundary = 1/3
        # left lane line segment should be on left 2/3 of the screen
        left_region_boundary = width * (1 - boundary)
         # right lane line segment should be on left 2/3 of the screen
        right_region_boundary = width * boundary

        for line_segment in line_segments:
            for x1, y1, x2, y2 in line_segment:
                if x1 == x2:
                    continue
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope < 0:
                    if x1 < left_region_boundary and x2 < left_region_boundary:
                        left_fit.append((slope, intercept))
                else:
                    if x1 > right_region_boundary and x2 > right_region_boundary:
                        right_fit.append((slope, intercept))

        left_fit_average = np.average(left_fit, axis=0)
        if len(left_fit) > 0:
            lane_lines.append(
                GenericLineDetector.make_points(frame, left_fit_average))

        right_fit_average = np.average(right_fit, axis=0)
        if len(right_fit) > 0:
            lane_lines.append(
                GenericLineDetector.make_points(frame, right_fit_average))

        return lane_lines

    def pipeline(self):
        if self.frame is None:
            raise ValueError("No image to process")

        denoised_frame =  GenericLineDetector.apply_blur(self._frame)

        # Turn frame to grayscale
        gray = cv2.cvtColor(denoised_frame, cv2.COLOR_BGR2GRAY)

        # Get all edges using Canny edge algorithm.
        edges = cv2.Canny(
            gray,
            self.canny_param.threshold_lower,
            self.canny_param.threshold_upper)

        # Create a masking trapezoid to get only edges in the area of interest
        cropped_edges = GenericLineDetector.region_of_interest(
            edges,
            self.roi.to_polygon(
                height=self.frame.shape[0], width=self.frame.shape[1])
        )

        self._frame_canny = cropped_edges

        # Get all line segments using Probabilistic Hough algorithm
        line_segments = cv2.HoughLinesP(
            cropped_edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_param.threshold,
            minLineLength=self.hough_param.min_line_length,
            maxLineGap=self.hough_param.max_line_gap)

        self._segments = line_segments

        if line_segments is not None:
            mask = np.zeros_like(self._frame)
            for line in line_segments:
                for x1, y1, x2, y2 in line:
                    cv2.line(mask, (x1,y1), (x2,y2), (0,255,0), 4)
            self._frame_segment = cv2.addWeighted(self._frame, 0.8, mask, 1 ,1)

        lane_segments =  GenericLineDetector.average_slope_intercept(
            self._frame, line_segments)

        self._lanes = lane_segments

        if lane_segments is not None:
            mask = np.zeros_like(self._frame)
            for lane in lane_segments:
                for x1, y1, x2, y2 in lane:
                    cv2.line(mask, (x1,y1), (x2,y2), (0,255,0), 4)
            self._frame_lanes = cv2.addWeighted(self._frame, 0.8, mask, 1 ,1)


def __test(image_path: str):
    canny_param = CannyParameters(
        threshold_lower=50,
        threshold_upper=150,
        aperture_size=3)

    hough_param = HoughParameters(
        threshold=20,
        min_line_length=50,
        max_line_gap=30)

    line_detector = GenericLineDetector(
        canny_param, hough_param, RegionOfInterest())

    line_detector.frame = cv2.imread(image_path)

    line_detector.pipeline()

    cv2.imshow('Canny edges', line_detector.frame_canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    result = np.concatenate(
        (line_detector.frame_segment, line_detector.frame_lanes), axis=1)

    cv2.imshow('Hough', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    __test(sys.argv[1])
