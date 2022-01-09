#!/usr/bin/env python3
import threading
from typing import Optional

import numpy as np
from flask import Flask, render_template
from pyroombaadapter import PyRoombaAdapter, PacketType, OIMode
from roomba_functions import stop_moving_for_bump_or_wheel_drop, get_line_follower_thread


# Globals
app = Flask(__name__)

# I guess PyRoombaAdapter expects radians per second?
ROTATION_SPEED = np.radians(10)
SPEED_MS = 0.15

# Intended to be run on a Raspberry Pi Zero
roomba = PyRoombaAdapter("/dev/ttyS0")
lock = threading.Lock()
stop_moving_for_bump_or_wheel_drop(roomba, lock)

line_follower_thread: Optional[threading.Thread] = None
line_follower_stop_signal: threading.Event


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/button/<button>', methods=['POST'])
def button(button):
    forward_speed = 0.
    rotation_speed = 0.

    if button == 'up':
        forward_speed = SPEED_MS
    elif button == 'down':
        forward_speed = -SPEED_MS
    elif button == 'left':
        rotation_speed = ROTATION_SPEED
    elif button == 'right':
        rotation_speed = -ROTATION_SPEED

    elif button == 'line_follow_start':
        global line_follower_thread, line_follower_stop_signal
        line_follower_thread, line_follower_stop_signal = get_line_follower_thread(roomba, lock)
        line_follower_thread.start()

    elif button == 'line_follow_stop':
        global line_follower_thread, line_follower_stop_signal
        if line_follower_thread is not None:
            line_follower_stop_signal.set()
            line_follower_thread.join()
            line_follower_thread = None

    with lock:
        roomba.change_mode_to_safe()
        roomba.move(forward_speed, rotation_speed)

    return 'ok'
