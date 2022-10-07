#!/usr/bin/env python3

from abc import ABCMeta, abstractmethod
from collections import namedtuple

# import duckietown_code_utils as dtu

__all__ = ["LineDetectorInterface", "Detections", "FAMILY_LINE_DETECTOR"]

FAMILY_LINE_DETECTOR = "line_detector"

Detections = namedtuple("Detections", ["lines", "normals", "area", "centers"])


class LineDetectorInterface(metaclass=ABCMeta):
    @abstractmethod
    def setImage(self, bgr):
        pass

    @abstractmethod
    def detectLines(self, color):
        """Returns a tuple of class Detections"""
