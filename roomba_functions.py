from typing import Callable, Tuple
from threading import Thread, Lock, Event
from pyroombaadapter import PyRoombaAdapter, PacketType
import time

from picamera import PiCamera

from line_follower import follow_line


def create_sensor_event(roomba: PyRoombaAdapter,
                        packet_types,
                        callback: Callable,
                        lock: Lock,
                        check_interval_s: float) -> Thread:
    """Spins up a thread to check a sensor value at a regular interval."""

    def func():
        while True:
            with lock:
                result = roomba.request_sensor_data(packet_types)
            callback(result)
            time.sleep(check_interval_s)

    t = Thread(target=func, daemon=True)
    t.start()
    return t


def stop_moving_for_bump_or_wheel_drop(roomba: PyRoombaAdapter, lock: Lock):
    def callback(results):
        wheel_bump = results[0]

        if not any(wheel_bump):
            return

        # Stop moving forward
        with lock:
            roomba.move(0, 0)

        bump = wheel_bump[0] or wheel_bump[1]  # Left or right
        wheel_drop = wheel_bump[2] or wheel_bump[3]
        if bump:
            print('Bump!')
        if wheel_drop:
            print("Wheel drop!")

        if not wheel_drop:
            # Reverse a little bit
            with lock:
                roomba.move(-0.1, 0)
                time.sleep(0.1)
                roomba.move(0, 0)

    packets = [PacketType.BUMP_AND_WHEEL_DROP]
    create_sensor_event(roomba, packets, callback, lock, 0.1)


def get_line_follower_thread(roomba: PyRoombaAdapter,
                             lock: Lock,
                             update_rate_hz=10) -> Tuple[Thread, Event]:
    kill_sig = Event()
    camera = PiCamera(resolution=(640, 480))
    camera.exposure_mode = 'sports'

    def func():
        interval_time = 1 / update_rate_hz
        last_time = time.time()
        while True:
            if kill_sig.is_set():
                camera.close()
                break

            with lock:
                follow_line(roomba, camera)

            # Sleep away the extra time
            elapsed = time.time() - last_time
            if elapsed < interval_time:
                time.sleep(interval_time - elapsed)
            else:
                print('get_line_follower_thread fell behind by', elapsed - interval_time)
            last_time += interval_time

    t = Thread(target=func, daemon=True)
    t.start()
    return t, kill_sig
