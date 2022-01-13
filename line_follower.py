from pyroombaadapter import PyRoombaAdapter
from line_follower_functions import detect_line


# These are variables that control how fast the robot moves.
# ROTATION_SPEED = np.radians(10)  # in degrees per second
FORWARD_SPEED = 0.1 * 1000  # in millimeters per second


# This defines a function that will run 10 times per second.
def follow_line(roomba, camera):

    # These are helper functions that tell the robot what to do.
    def spin():
        print('spin')
        roomba.send_drive_direct(FORWARD_SPEED / 2, -FORWARD_SPEED / 2)

    def turn_right():
        print('turn_right')
        roomba.send_drive_direct(FORWARD_SPEED * 0.25, FORWARD_SPEED * 0.75)

    def turn_left():
        print('turn_left')
        roomba.send_drive_direct(FORWARD_SPEED * 0.75, FORWARD_SPEED * 0.25)

    def go_straight():
        print('go_straight')
        roomba.send_drive_direct(FORWARD_SPEED, FORWARD_SPEED)

    # This statement looks for a line in images taken from the camera.
    line_found, line_position = detect_line(camera, thresh=5)
    # return

    # If the line was not found, tell the robot to spin. Then, exit the function.
    if not line_found:
        spin()
        return

    # If the line is to the right of the robot, tell the robot to turn right.
    if line_position == "right":
        turn_right()

    # If the line is to the left of the robot, tell the robot to turn left.
    if line_position == "left":
        turn_left()

    # If the line is straight ahead, tell the robot to go straight.
    if line_position == "center":
        go_straight()

    # Exit the function
    return


def follow_line2(roomba, camera):
    line_found, line_position = detect_line(camera, string_result=False)

    if not line_found:
        # Spin in place and look for the line
        roomba.send_drive_direct(FORWARD_SPEED / 2, -FORWARD_SPEED / 2)
        return

    alpha = (line_position + 10) / 20
    roomba.send_drive_direct(FORWARD_SPEED * (1-alpha), FORWARD_SPEED * alpha)

    return
