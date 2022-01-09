from pyroombaadapter import PyRoombaAdapter
from line_follower_functions import detect_line


# These are variables that control how fast the robot moves.
FORWARD_SPEED = 0.15  # in meters per second
ROTATION_SPEED = 10  # in degrees per second


# This defines a function that will run 10 times per second.
def follow_line(roomba, camera):

    # These are helper functions that tell the robot what to do.
    def spin():
        roomba.move(0, ROTATION_SPEED)

    def turn_right():
        roomba.move(FORWARD_SPEED, ROTATION_SPEED)

    def turn_left():
        roomba.move(FORWARD_SPEED, -ROTATION_SPEED)

    def go_straight():
        roomba.move(FORWARD_SPEED, 0)

    # This looks for a line in images taken from the camera.
    line_found, line_position = detect_line(camera)

    # If the line was not found, tell the robot to spin. Then, exit the function.
    if not line_found:
        spin()
        return

    # If the line is to the right of the robot, tell it to turn right.
    if line_position == 10:
        turn_right()

    # If the line is to the left of the robot, tell it to turn left.
    if line_position == -10:
        turn_left()

    # If the line is straight ahead, tell it to go straight.
    if line_position == 0:
        go_straight()

    # Exit the function
    return


def follow_line_advanced(roomba, camera):
    line_found, line_position = detect_line(camera)

    if not line_found:
        # Spin in place and look for the line
        roomba.move(0, ROTATION_SPEED)
        return

    reduced_rotation_speed = ROTATION_SPEED / 10
    proportional_rotation_speed = reduced_rotation_speed * line_position
    roomba.move(FORWARD_SPEED, proportional_rotation_speed)

    return
