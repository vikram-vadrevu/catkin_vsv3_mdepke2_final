#!/usr/bin/env python

import sys
import copy
import time
import rospy

import numpy as np
from final_header import *
from final_func import *
from image_proc import *

################ Pre-defined parameters and functions below (can change if needed) ################

# 20Hz
SPIN_RATE = 20  

# UR3 home location
home = [270*PI/180.0, -90*PI/180.0, 90*PI/180.0, -90*PI/180.0, -90*PI/180.0, 135*PI/180.0]  

height = 0.2

# UR3 current position, using home position for initialization
current_position = copy.deepcopy(home)  

thetas = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

digital_in_0 = 0
analog_in_0 = 0.0

suction_on = True
suction_off = False

current_io_0 = False
current_position_set = False


"""
Whenever ur3/gripper_input publishes info this callback function is called.
"""
def input_callback(msg):
    global digital_in_0
    digital_in_0 = msg.DIGIN
    digital_in_0 = digital_in_0 & 1 # Only look at least significant bit, meaning index 0


"""
Whenever ur3/position publishes info, this callback function is called.
"""
def position_callback(msg):
    global thetas
    global current_position
    global current_position_set

    thetas[0] = msg.position[0]
    thetas[1] = msg.position[1]
    thetas[2] = msg.position[2]
    thetas[3] = msg.position[3]
    thetas[4] = msg.position[4]
    thetas[5] = msg.position[5]

    current_position[0] = thetas[0]
    current_position[1] = thetas[1]
    current_position[2] = thetas[2]
    current_position[3] = thetas[3]
    current_position[4] = thetas[4]
    current_position[5] = thetas[5]

    current_position_set = True


"""
Function to control the suction cup on/off
"""
def gripper(pub_cmd, loop_rate, io_0):
    global SPIN_RATE
    global thetas
    global current_io_0
    global current_position

    error = 0
    spin_count = 0
    at_goal = 0

    current_io_0 = io_0

    driver_msg = command()
    driver_msg.destination = current_position
    driver_msg.v = 1.0
    driver_msg.a = 1.0
    driver_msg.io_0 = io_0
    pub_cmd.publish(driver_msg)

    while(at_goal == 0):

        if( abs(thetas[0]-driver_msg.destination[0]) < 0.0005 and \
            abs(thetas[1]-driver_msg.destination[1]) < 0.0005 and \
            abs(thetas[2]-driver_msg.destination[2]) < 0.0005 and \
            abs(thetas[3]-driver_msg.destination[3]) < 0.0005 and \
            abs(thetas[4]-driver_msg.destination[4]) < 0.0005 and \
            abs(thetas[5]-driver_msg.destination[5]) < 0.0005 ):

            rospy.loginfo("Goal is reached!")
            at_goal = 1

        loop_rate.sleep()

        if(spin_count >  SPIN_RATE*5):

            pub_cmd.publish(driver_msg)
            rospy.loginfo("Just published again driver_msg")
            spin_count = 0

        spin_count = spin_count + 1

    return error


"""
Move robot arm from one position to another
"""
def move_arm(pub_cmd, loop_rate, dest, vel, accel, move_type):
    global thetas
    global SPIN_RATE

    error = 0
    spin_count = 0
    at_goal = 0

    driver_msg = command()
    driver_msg.destination = dest
    driver_msg.v = vel
    driver_msg.a = accel
    driver_msg.io_0 = current_io_0
    driver_msg.move_type = move_type  # Move type (MoveJ or MoveL)
    pub_cmd.publish(driver_msg)

    loop_rate.sleep()

    while(at_goal == 0):

        if( abs(thetas[0]-driver_msg.destination[0]) < 0.0005 and \
            abs(thetas[1]-driver_msg.destination[1]) < 0.0005 and \
            abs(thetas[2]-driver_msg.destination[2]) < 0.0005 and \
            abs(thetas[3]-driver_msg.destination[3]) < 0.0005 and \
            abs(thetas[4]-driver_msg.destination[4]) < 0.0005 and \
            abs(thetas[5]-driver_msg.destination[5]) < 0.0005 ):

            at_goal = 1
            rospy.loginfo("Goal is reached!")

        loop_rate.sleep()

        if(spin_count >  SPIN_RATE*5):

            pub_cmd.publish(driver_msg)
            rospy.loginfo("Just published again driver_msg")
            spin_count = 0

        spin_count = spin_count + 1

    return error

################ Pre-defined parameters and functions above (can change if needed) ################

##========= TODO: Helper Functions =========##

def find_path(image_path):
    """Gets keypoints from the given image

    Parameters
    ----------
    image_path : String
        The given image (before or after preprocessing)

    Returns
    -------
    keypoints
        a list of keypoints detected in image coordinates
    """
    feature_extractor = ImaceProc(image_path)

    contours = feature_extractor.get_contours()

    line_segments = feature_extractor.approximate_splines(contours, 0.001)

    return line_segments



def IMG2W(row, col, image):
    """Transform image coordinates to world coordinates

    Parameters
    ----------
    row : int
        Pixel row position
    col : int
        Pixel column position
    image : np.ndarray
        The given image (before or after preprocessing)

    Returns
    -------
    x : float
        x position in the world frame
    y : float
        y position in the world frame
    """
    x, y = 0.0, 0.0
    return (x, y)

def compute_optimal_path(lines):
    """Compute the optimal path with the fewest discontinuities

    Parameters
    ----------
    lines : list
        List of line segments, where each segment is represented as a tuple of start and end points

    Returns
    -------
    optimal_path : list
        List of lists where each element represents a continuous set of line segments
    """
    if not lines:
        return []

    print(f'Computing optimal path for {len(lines)} line segments. This might take a while...')

    start_time = time.time()

    # Initialize the optimal path with the first line segment
    optimal_path = [[lines[0]]]
    lines = lines[1:]

    while lines:
        last_segment = optimal_path[-1][-1]
        last_point = last_segment[1]
        min_distance = float('inf')
        next_segment = None
        next_index = -1

        # Find the closest segment to the last point in the current path
        for i, segment in enumerate(lines):
            start_point = segment[0]
            distance = np.linalg.norm(np.array(last_point) - np.array(start_point))
            if distance < min_distance:
                min_distance = distance
                next_segment = segment
                next_index = i

        # If the closest segment is continuous, add it to the current path
        if min_distance < 0.001:
            optimal_path[-1].append(next_segment)
        else:
            # Otherwise, start a new path
            optimal_path.append([next_segment])

        # Remove the selected segment from the list
        lines.pop(next_index)

    print(f'Optimal path computed in {time.time() - start_time:.2f} seconds')

    return optimal_path


def draw_image(path, pub_command, loop_rate):
    """Draw the image based on detecte keypoints in world coordinates

    Parameters
    ----------
    path:
        a list of keypoints detected in world coordinates
    """

    optimal_pixel_path = compute_optimal_path(path)

    for i, segment in enumerate(optimal_pixel_path):
        for line in segment:
            start, end = line
            print("Drawing line from", start, "to", end)
            joint_position = lab_invk(end[0], end[1], height, 0.0)
            move_arm(pub_command, loop_rate, joint_position, 1, 1, 'L')
        
        current_pos = segment[-1][-1]
        joint_position = lab_invk(current_pos[0], current_pos[1] + 0.2, height, 0.0)
        move_arm(pub_command, loop_rate, joint_position, 1, 1, 'L')

        if i == len(optimal_pixel_path) - 1:
            break

        next_pos = optimal_pixel_path[i+1][0][0]
        joint_position = lab_invk(next_pos[0], next_pos[1] + 0.2, height, 0.0)
        move_arm(pub_command, loop_rate, joint_position, 1, 1, 'J')

        joint_position = lab_invk(next_pos[0], next_pos[1], height, 0.0)
        move_arm(pub_command, loop_rate, joint_position, 1, 1, 'L')
        time.sleep(0.3)


"""
Program run from here
"""
def main():
    global home
    # global variable1
    # global variable2

    # Initialize ROS node
    rospy.init_node('lab5node')

    # Initialize publisher for ur3/command with buffer size of 10
    pub_command = rospy.Publisher('ur3/command', command, queue_size=10)

    # Initialize subscriber to ur3/position & ur3/gripper_input and callback fuction
    # each time data is published
    sub_position = rospy.Subscriber('ur3/position', position, position_callback)
    sub_input = rospy.Subscriber('ur3/gripper_input', gripper_input, input_callback)

    # Check if ROS is ready for operation
    while(rospy.is_shutdown()):
        print("ROS is shutdown!")

    # Initialize the rate to publish to ur3/command
    loop_rate = rospy.Rate(SPIN_RATE)

    # Velocity and acceleration of the UR3 arm
    vel = 4.0
    accel = 4.0
    move_arm(pub_command, loop_rate, home, vel, accel, 'J')  # Move to the home position

    ##========= TODO: Read and draw a given image =========##
    paper_offset = np.array([16, 15, 1.5])/100
    image_path = 'images/zigzag.jpg'
    

    start = lab_invk(paper_offset[0], paper_offset[1], paper_offset[2], 0)
    move_arm(pub_command, loop_rate, start, 3, 3, 'J')

    time.sleep(2)

    proc = ImageProc(image_path)
    points = proc.get_lines()
    scaled = proc.scale_to_meters(points)

    for contour in scaled:
        for point in contour:
            print("point: ", type(point))
            pos = np.array([point[0], point[1], 0])
            print(pos)
            pos = paper_offset + pos

            angles = lab_invk(pos[0], pos[1], pos[2], 0.)
            move_arm(pub_command, loop_rate, angles, 1, 1, 'L')
            time.sleep(0.1)


    move_arm(pub_command, loop_rate, home, vel, accel, 'J')  # Return to the home position
    rospy.loginfo("Task Completed!")
    print("Use Ctrl+C to exit program")
    rospy.spin()

if __name__ == '__main__':

    try:
        main()
    # When Ctrl+C is executed, it catches the exception
    except rospy.ROSInterruptException:
        pass
