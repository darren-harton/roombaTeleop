import io
import time
from typing import List
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2

import picamera
import picamera.array
from picamera import PiCamera


from picamera import mmal, mmalobj as mo
from threading import Thread, Lock


ENABLE_STRING_RESULT = True
FRAME_NUM = 0
IMG_DIR = Path('.') / "test_images"


class ImageBuffer:
    def __init__(self):
        camera = mo.MMALCamera()

        camera.outputs[0].framesize = (640 // 4, 480 // 4)
        camera.outputs[0].framerate = 30
        camera.outputs[0].format = mmal.MMAL_ENCODING_RGB24
        camera.outputs[0].commit()

        camera.outputs[0].enable(self.image_callback)
        self.camera = camera
        self.lock = Lock()
        self.image = None

    def __del__(self):
        self.camera.outputs[0].disable()

    def image_callback(self, port, buf):
        with self.lock:
            self.image = buf.data

    def get_image(self):
        with self.lock:
            return np.array(self.image, copy=True)


# TODO: broken :(
# IMAGE_BUFFER = ImageBuffer()


def capture_image(camera, crop=None):
    start = time.time()

    with picamera.array.PiRGBArray(camera) as stream:
        camera.capture(stream, format='bgr')
        # print('capture', time.time() - start)
        # At this point the image is available as stream.array
        image = stream.array

    # image = IMAGE_BUFFER.get_image()
    # print('capture', time.time() - start)

    if crop is None:
        return image

    # crop image to rectangle
    x1, y1, x2, y2 = crop
    roi = image[y1:y2+1, x1:x2+1]
    return roi

def capture_image_old(camera, crop=None):
    start = time.time()

    # Create the in-memory stream
    stream = io.BytesIO()
    camera.capture(stream, format='jpeg')
    print('capture', time.time() - start)

    # Construct a numpy array from the stream
    data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
    print('frombuffer', time.time() - start)
    # convert to OpenCV Mat
    image = cv2.imdecode(data, 1)
    print('imdecode', time.time() - start)
    # Convert from BGR to RGB
    image = image[:, :, ::-1]
    print('flip', time.time() - start)

    # image = undistort(image)  # TODO if needed

    if crop is None:
        return image

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


def get_hough_lines(img, rho=1, theta=np.pi/500, threshold=20, min_line_len=20, max_line_gap=300):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None:
        return lines
    if np.ndim(lines) == 3:
        lines = lines.squeeze()

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
    img = np.float32(img)
    for line in lines:
        x1, y1, x2, y2 = [int(x) for x in line]
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    return img


def get_line_slope_intercept(line):
    x1, y1, x2, y2 = line
    if x2-x1 == 0:
        return np.inf, 0
    slope = (y2-y1)/(x2-x1)
    intercept = y1 - slope * x1
    return slope, intercept


def apply_binning(point, img_size, num_bins=21):
    x, y = point
    h, w = img_size
    bins = np.linspace(0, w, num_bins)
    # Get the appropriate bin using the point's `y` value.
    bin_idx = np.digitize([x], bins) - (num_bins//2 + 1)
    return bin_idx


def sort_lines(lines):
    """Sort lines so they are going mostly the same direction"""
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line
        pt1 = (x1, y1)
        pt2 = (x2, y2)
        if x1 > x2 or y1 > y2:
            pt1, pt2 = pt2, pt1
        lines[i] = pt1 + pt2
    return lines


def save_img(img, name):
    global FRAME_NUM
    bgr = img[:, :, ::-1] if img.shape[-1] == 3 else img
    cv2.imwrite(str(IMG_DIR / f'{FRAME_NUM:04}_{name}.jpg'), bgr)


def detect_line(camera: PiCamera, thresh=4, debug=False, string_result=True):
    try:
        return _detect_line(camera, thresh, debug, string_result)
    except Exception as e:
        print("ERROR:", e)
    return False, None


def _detect_line(camera: PiCamera, thresh, debug, string_result):
    start = time.time()

    original_img = capture_image(camera)
    # print('got image', time.time() - start)

    # Create a working copy
    img = np.copy(original_img)
    # print('copy', time.time() - start)
    if debug:
        global FRAME_NUM
        FRAME_NUM += 1

        IMG_DIR.mkdir(parents=True, exist_ok=True)
        save_img(img, "original")

    # Isolate blue in the image
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # lower_blue = np.array([60, 35, 140])
    lower_blue = np.array([60, 35, 50])
    # upper_blue = np.array([200, 255, 255])
    upper_blue = np.array([200, 255, 255])
    blue_mask = cv2.inRange(hls_img, lower_blue, upper_blue)
    # print('blue_mask', time.time() - start)

    img = cv2.bitwise_and(img, img, mask=blue_mask)
    # print('bitwise_and', time.time() - start)
    # img[~blue_mask] = 0

    if not img.any():
        print("masked image is empty.")
        return False, None

    if debug:
        save_img(img, "masked")

    # Apply Gaussian Blur to reduce noise
    img = cv2.GaussianBlur(img, (7, 7), 0)
    if debug:
        save_img(img, "blurred")

    # Apply Canny edge filter
    img = cv2.Canny(img, 70, 140)
    if debug:
        save_img(img, "canny")

    # Get region of interest
    # TODO: find good values for this depending on mounting and FOV
    # img = crop(img)

    # Apply Hough lines
    hough_lines = get_hough_lines(img)
    if hough_lines is None or len(hough_lines) == 0:
        return False, None

    hough_lines = sort_lines(hough_lines)

    if debug:
        hough_img = draw_lines(original_img, hough_lines)
        save_img(hough_img, "hough")

    line = average_lines(hough_lines)
    if debug:
        avg_line = draw_lines(original_img, [line])
        save_img(avg_line, "avg_line")

    slope, _ = get_line_slope_intercept(line)

    horizontal_thresh = np.radians(10)
    # Check if the line appears horizontal-ish
    if abs(slope) < horizontal_thresh:
        # Get the middle of the line
        target_point = get_line_midpoint(line)
    else:
        # Pick whichever point is lower in the image. Remember image coords mean higher Y is lower in the image.
        x1, y1, x2, y2 = line
        pt1 = (x1, y1)
        pt2 = (x2, y2)
        # target_point = pt2 if pt1[1] < pt2[1] else pt1
        target_point = get_line_midpoint(line)

    if debug:
        target_img = draw_lines(original_img, [(target_point + target_point)], thickness=5)
        save_img(target_img, "target_img")

    line_pos = apply_binning(target_point, img.shape)
    print('line_pos', line_pos)

    if string_result:
        if 0 < line_pos and thresh < line_pos:
            line_pos = "right"
        elif 0 > line_pos and thresh > line_pos:
            line_pos = "left"
        else:
            line_pos = "center"

        print('Moving', line_pos)

    return True, line_pos


