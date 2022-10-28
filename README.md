
## Packages

### Lane Following

1. lane_detection/line_detector
    
Detect the line white, yellow and red line segment in an image and is used for
lane localization. Subscribes to the input video stream.

2. lane_filter

Generates an estimate of the lane pose. Get estimate on `d` and `phi`, the
lateral and heading deviation from the center of the lane. It gets the segments
extracted by the line_detector as input and output the lane pose estimate.

3. ground_projection

Projects the line segments detected in the image to the ground plane and in the
robot's reference frame. In this way it enables lane localization in the 2D
ground plane. This projection is performed using the homography matrix obtained
from the extrinsic calibration procedure.

4. stop_line_filter

Detect red stop line detected line segments.

5. lane_controller

Computes the commands in form of linear and angular velocities, by processing
the estimate error in lateral deviationa and heading.

6. anti_instagram

Make lane detection robust to illumination variation.
More information can be found [here](https://github.com/duckietown/docs-fall2017_projects/blob/master/book/fall2017_projects/27_anti_instagram/30-final-project-report-anti-instagram.md).
