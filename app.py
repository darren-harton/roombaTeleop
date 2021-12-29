#!/usr/bin/env python3

from flask import Flask, render_template, request
app = Flask(__name__)

import numpy as np
from pyroombaadapter import PyRoombaAdapter


# I guess PyRoombaAdapter expects radians per second?
ROTATION_SPEED = np.radians(10)
SPEED_MS = 0.15


# Intended to be run on a Raspberry Pi Zero
roomba = PyRoombaAdapter("/dev/ttyS0")


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

    roomba.move(forward_speed, rotation_speed)

    return 'ok'
