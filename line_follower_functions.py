import io
from typing import List
from dataclasses import dataclass

import numpy as np
import cv2

from picamera import PiCamera


@dataclass
class Line:
    all_pix: List[float]
    best_fit: List[float]  # a and b from `y = ax + b`


"""Skip calibration stuff for now"""
# def load_camera_cal(cal_path):
#     # TODO
#     raise NotImplementedError()
#
# camera_matrix = load_camera_cal("/home/pi/camera_cal.json")


def capture_image(camera, crop=(0, 0, -1, -1)):
    # Create the in-memory stream
    stream = io.BytesIO()
    camera.capture(stream, format='jpeg')

    # Construct a numpy array from the stream
    data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
    # convert to OpenCV Mat
    image = cv2.imdecode(data, 1)
    # Convert from BGR to RGB
    image = image[:, :, ::-1]

    # image = undistort(image)  # TODO if needed

    # crop image to rectangle
    x1, y1, x2, y2 = crop
    roi = image[y1:y2+1, x1:x2+1]
    return roi


def crop(img, corner1, corner2):
    x1, y1 = corner1
    x2, y2 = corner2

    roi = img[y1:y2 + 1, x1:x2 + 1]
    return roi

def get_threshold_values(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]  # Luminance
    s_channel = hls[:, :, 2]  # Saturation

    # Apply Sobel filter on the X axis
    # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    sobel = np.abs(cv2.Sobel(l_channel, cv2.CV_64F, 1, 0))
    # Normalize to 0-255
    sobel = np.uint8(255 * sobel / np.max(sobel))

    # Apply threshold values to hopefully filter noise
    sxbinary = np.zeros_like(sobel)
    sxbinary[(sobel >= sx_thresh[0]) & (sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary


def get_hough_lines(img, rho=1, theta=np.pi/180, threshold=20, min_line_len=20, max_line_gap=300):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


def average_lines(lines):
    """Weighted average based on length"""
    lengths = np.linalg.norm(lines, axis=1)
    avg = np.dot(lengths, lines) / np.sum(lengths)
    return avg


def get_line_midpoint(line):
    x1, y1, x2, y2 = line
    return (x1+x2)/2, (y1+y2)/2


def draw_lines(img, lines, color=(255, 0, 0), thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    return img


def get_line_slope_intercept(line):
    x1, y1, x2, y2 = line
    if x2-x1 == 0:
        return np.inf, 0
    slope = (y2-y1)/(x2-x1)
    intercept = y1 - slope * x1
    return slope, intercept


def detect_line(camera: PiCamera):
    success = False
    line_pos = 0

    original_img = capture_image(camera)

    # Create a working copy
    img = np.copy(original_img)

    # Isolate blue in the image
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    low_thresh = np.array([0, 0, 50], np.uint8)
    high_thresh = np.array([0, 0, 255], np.uint8)
    blue_mask = cv2.inRange(hls_img, low_thresh, high_thresh)
    img = img[blue_mask]

    # Apply Gaussian Blur to reduce noise
    img = cv2.GaussianBlur(img, (7, 7), 0)

    # Apply Canny edge filter
    img = cv2.Canny(img, 70, 140)

    # Get region of interest
    # TODO: find good values for this depending on mounting and FOV
    # img = crop(img)

    # Apply Hough lines
    hough_lines = get_hough_lines(img)
    # hough_img = draw_lines(original_img, hough_lines)

    line = average_lines(hough_lines)
    slope, _ = get_line_slope_intercept(line)

    line_value = None

    horizontal_thresh = np.radians(10)
    if abs(slope) < horizontal_thresh:
        mid = get_line_midpoint(line)
        line_value = get_line_midpoint(line)
    else:
        line_value

    # Extrapolation and averaging
    left_lane, right_lane = get_lane_lines(original_img, hough_lines)

    return success, line_pos


