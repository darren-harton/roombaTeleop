#!/usr/bin/env python3
import threading
from typing import Optional

import numpy as np
from flask import Flask, render_template, Response
from pyroombaadapter import PyRoombaAdapter, PacketType, OIMode
from roomba_functions import stop_moving_for_bump_or_wheel_drop, get_line_follower_thread

from picamera import PiCamera
import io
import cv2


# Globals
app = Flask(__name__)


ROTATION_SPEED = np.radians(10)  # in degrees per second
FORWARD_SPEED = 0.15  # meters per second

# Intended to be run on a Raspberry Pi Zero
roomba = PyRoombaAdapter("/dev/ttyS0")
lock = threading.Lock()
stop_moving_for_bump_or_wheel_drop(roomba, lock)


line_follower_thread: Optional[threading.Thread] = None
line_follower_stop_signal: threading.Event = None

camera = PiCamera(resolution=(640 // 4, 480 // 4), framerate=10)
camera.exposure_mode = 'sports'
camera.vflip = True
camera.hflip = True


def capture_image(camera, crop=None):
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

    if crop is None:
        return image

    # crop image to rectangle
    x1, y1, x2, y2 = crop
    roi = image[y1:y2+1, x1:x2+1]
    return roi


def gen_frames():
    while True:
        stream = io.BytesIO()
        camera.capture(stream, format='jpeg')
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + stream.getvalue() + b'\r\n')  # concat frame one by one and show result
        break



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/button/<button>', methods=['POST'])
def button(button):
    forward_speed = 0.
    rotation_speed = 0.

    global line_follower_thread
    global line_follower_stop_signal

    if button == 'up':
        forward_speed = FORWARD_SPEED
    elif button == 'down':
        forward_speed = -FORWARD_SPEED
    elif button == 'left':
        rotation_speed = ROTATION_SPEED
    elif button == 'right':
        rotation_speed = -ROTATION_SPEED

    elif button == 'line_follow_start':
        if line_follower_thread is None:
            line_follower_thread, line_follower_stop_signal = get_line_follower_thread(roomba, camera, lock)

    elif button == 'line_follow_stop':
        if line_follower_thread is not None:
            line_follower_stop_signal.set()
            line_follower_thread.join()
            line_follower_thread = None

    with lock:
        roomba.change_mode_to_safe()
        print("Moving", forward_speed, rotation_speed)
        roomba.move(forward_speed, rotation_speed)

    return 'ok'


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
