#!/usr/bin/env python3

import cv2
import numpy as np

from dt_image_processing_utils import find_normal

from line_detector.include.detections import Detections
from line_detector.include.line_detector_interface import LineDetectorInterface


__all__ = ["LineDetector"]


class LineDetector(LineDetectorInterface):
    def __init__(
        self,
        canny_thresholds=[80, 200],
        canny_aperture_size=3,
        dilation_kernel_size=3,
        hough_threshold=2,
        hough_min_line_length=3,
        hough_max_line_gap=1,
    ):

        self.canny_thresholds = canny_thresholds
        self.canny_aperture_size = canny_aperture_size
        self.dilation_kernel_size = dilation_kernel_size
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap

        # initialize the variables that will hold the processed images
        self.bgr = np.empty(0)  #: Holds the ``BGR`` representation of an image
        self.hsv = np.empty(0)  #: Holds the ``HSV`` representation of an image
        self.canny_edges = np.empty(0)  #: Holds the Canny edges of an image

    def setImage(self, image):
        """Sets the :py:attr:`bgr` attribute to the provided image. Also stores
        an `HSV <https://en.wikipedia.org/wiki/HSL_and_HSV>`_
        representation of the image and the extracted
        `Canny edges <https://en.wikipedia.org/wiki/Canny_edge_detector>`_.
        This is separated from :py:meth:`detectLines` so that the HSV
        representation and the edge extraction can be reused for multiple
        colors.
        Args:
            image (:obj:`numpy array`): input image
        """

        self.bgr = np.copy(image)
        self.hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.canny_edges = self.findEdges()

    def getImage(self):
        """
        Provides the image currently stored in the :py:attr:`bgr` attribute.
        Returns:
            :obj:`numpy array`: the stored image
        """
        return self.bgr

    def findEdges(self):
        """ Applies `Canny edge detection
        <https://en.wikipedia.org/wiki/Canny_edge_detector>`_ to
        a ``BGR`` image.
        Returns:
            :obj:`numpy array`: a binary image with the edges
        """
        edges = cv2.Canny(
            self.bgr,
            self.canny_thresholds[0],
            self.canny_thresholds[1],
            apertureSize=self.canny_aperture_size,
        )
        return edges

    def houghLine(self, edges):
        """ Finds line segments in a binary image using the probabilistic
        Hough transform. Based on the OpenCV function
        `HoughLinesP <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghlinesp
        #houghlinesp>`_.
        Args:
            edges (:obj:`numpy array`): binary image with edges
        Returns:
             :obj:`numpy array`: An ``Nx4`` array where each row represents a
                line ``[x1, y1, x2, y2]``. If no lines were detected,
                returns an empty list.
        """
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )
        if lines is not None:
            lines = lines.reshape((-1, 4))  # it has an extra dimension
        else:
            lines = []

        return lines

    def colorFilter(self, color_range):
        """Obtains the regions of the image that fall in the provided color
        range and the subset of the detected Canny edges which are in these
        regions. Applies a `dilation <https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm>`_
        operation to smooth and grow the regions map.
        Args:
            color_range (:py:class:`ColorRange`): A :py:class:`ColorRange`
                object specifying the desired colors.
        Returns:
            :obj:`numpy array`: binary image with the regions of the image
                that fall in the color range
            :obj:`numpy array`: binary image with the edges in the image that
                fall in the color range
        """
        # threshold colors in HSV space
        map = color_range.inRange(self.hsv)

        # binary dilation: fills in gaps and makes the detected regions grow
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.dilation_kernel_size, self.dilation_kernel_size)
        )
        map = cv2.dilate(map, kernel)

        # extract only the edges which come from the region with
        # the selected color
        edge_color = cv2.bitwise_and(map, self.canny_edges)

        return map, edge_color

    def detectLines(self, color_range):
        """Detects the line segments in the currently set image that occur in
        and the edges of the regions of the image that are within the provided
        colour ranges.
        Args:
            color_range (:py:class:`ColorRange`): A :py:class:`ColorRange`
                object specifying the desired colors.
        Returns:
            :py:class:`Detections`: A :py:class:`Detections` object with the
                map of regions containing the desired colors, and the
                detected lines, together with their center points and normals.
        """
        map, edge_color = self.colorFilter(color_range)
        lines = self.houghLine(edge_color)
        centers, normals = find_normal(map, lines)
        return Detections(lines=lines, normals=normals, map=map, centers=centers)
