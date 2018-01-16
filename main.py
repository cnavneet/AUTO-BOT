from robot import *
from math import *
from matrix import *
import random
from collections import deque


def robot_x_fn(state, dt=1.0):
    """
    State update for nonlinear system
    Computes next state using the non-linear dynamics
    """
    x = state.value[0][0]
    y = state.value[1][0]
    theta = state.value[2][0]
    v = state.value[3][0]
    w = state.value[4][0]

    x += v * cos(theta)*dt
    y += v * sin(theta)*dt
    theta += w*dt

    return matrix([[x],
                   [y],
                   [theta],
                   [v],
                   [w]])


def state_from_measurements(three_measurements):
    """
    Estimates state of robot from the last three measurements
    Assumes each movement of robot is a "step" and a "turn"
    Three measurements constitute two moves, from which turn angle, heading
    and step size can be inferred.
    """

    x1, y1 = three_measurements[-3]
    x2, y2 = three_measurements[-2]
    x3, y3 = three_measurements[-1]

    # Last two position vectors from measurements
    vec_1 = [x2 - x1, y2 - y1]
    vec_2 = [x3 - x2, y3 - y2]

    # Find last turning angle using dot product
    dot = sum(v1*v2 for v1,v2 in zip(vec_1, vec_2))
    mag_v1 = sqrt(sum(v**2 for v in vec_1))
    mag_v2 = sqrt(sum(v**2 for v in vec_2))

    v0 = mag_v2
    w0 = acos(dot/(mag_v1*mag_v2))
    if dot < 0:
        w0 = pi-w0
    theta0 = atan2(vec_2[1], vec_2[0]) + w0
    x0 = x3 + v0*cos(theta0 + w0)
    y0 = y3 + v0*sin(theta0 + w0)

    return matrix([[x3], [y3], [theta0], [v0], [w0]])


def next_move(hunter_position, hunter_heading, target_measurement, max_distance, OTHER = None):
    # This function will be called after each time the target moves.
    if OTHER is None: OTHER = deque(maxlen=3)
    OTHER.append(target_measurement)
    if len(OTHER) < 3:
        # Use naive tracking until target is localized
        heading_to_target = get_heading(hunter_position, target_measurement)
        heading_difference = heading_to_target - hunter_heading
        turning =  heading_difference # turn towards the target
        distance = max_distance # full speed ahead!
    else:
        # Estimate current state of robot
        state = state_from_measurements(OTHER)

        # Estimate number of steps required to reach target
        num_steps = 1
        while True:
            state = robot_x_fn(state)
            x, y = state.value[0][0], state.value[1][0]
            theta, v, w = state.value[2][0], state.value[3][0], state.value[4][0]
            est_target_pos = [x, y]
            separation = distance_between(est_target_pos, hunter_position)
            if separation < num_steps*max_distance:
                break

            num_steps += 1


        heading_to_target = get_heading(hunter_position, est_target_pos)
        heading_difference = heading_to_target - hunter_heading
        turning =  heading_difference # turn towards the target
        distance = distance_between(est_target_pos, hunter_position)
        distance = min(distance, max_distance)

    return turning, distance, OTHER


def naive_next_move(hunter_position, hunter_heading, target_measurement, max_distance, OTHER):
    """This strategy always tries to steer the hunter directly towards where the target last
    said it was and then moves forwards at full speed. This strategy also keeps track of all
    the target measurements, hunter positions, and hunter headings over time, but it doesn't
    do anything with that information."""
    if not OTHER: # first time calling this function, set up my OTHER variables.
        measurements = [target_measurement]
        hunter_positions = [hunter_position]
        hunter_headings = [hunter_heading]
        OTHER = (measurements, hunter_positions, hunter_headings) # now I can keep track of history
    else: # not the first time, update my history
        OTHER[0].append(target_measurement)
        OTHER[1].append(hunter_position)
        OTHER[2].append(hunter_heading)
        measurements, hunter_positions, hunter_headings = OTHER # now I can always refer to these variables

    heading_to_target = get_heading(hunter_position, target_measurement)
    heading_difference = heading_to_target - hunter_heading
    turning =  heading_difference # turn towards the target
    distance = max_distance # full speed ahead!
    return turning, distance, OTHER


def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def angle_trunc(a):
    """This maps all angles to a domain of [-pi, pi]"""
    while a < 0.0:
        a += pi * 2
    return ((a + pi) % (pi * 2)) - pi


def get_heading(hunter_position, target_position):
    """Returns the angle, in radians, between the target and hunter positions"""
    hunter_x, hunter_y = hunter_position
    target_x, target_y = target_position
    heading = atan2(target_y - hunter_y, target_x - hunter_x)
    # heading = angle_trunc(heading)
    heading = atan2(sin(heading), cos(heading))
    return heading


def demo_grading(hunter_bot, target_bot, next_move_fcn, OTHER = None):
    """Returns True if your next_move_fcn successfully guides the hunter_bot
    to the target_bot. This function is here to help you understand how we
    will grade your submission."""
    max_distance = 0.98 * target_bot.distance # 0.98 is an example. It will change.
    separation_tolerance = 0.02 * target_bot.distance # hunter must be within 0.02 step size to catch target
    caught = False
    ctr = 0

    # We will use your next_move_fcn until we catch the target or time expires.
    while not caught and ctr < 1000:

        # Check to see if the hunter has caught the target.
        hunter_position = (hunter_bot.x, hunter_bot.y)
        target_position = (target_bot.x, target_bot.y)
        separation = distance_between(hunter_position, target_position)
        if separation < separation_tolerance:
            print("You got it right! It took you ", ctr, " steps to catch the target.")
            caught = True

        # The target broadcasts its noisy measurement
        target_measurement = target_bot.sense()

        # This is where YOUR function will be called.
        turning, distance, OTHER = next_move_fcn(hunter_position, hunter_bot.heading, target_measurement, max_distance, OTHER)

        # Don't try to move faster than allowed!
        if distance > max_distance:
            distance = max_distance

        # We move the hunter according to your instructions
        hunter_bot.move(turning, distance)

        # The target continues its (nearly) circular motion.
        target_bot.move_in_circle()

        ctr += 1
        if ctr >= 1000:
            print("It took too many steps to catch the target.")
    return caught

def demo_grading2(hunter_bot, target_bot, next_move_fcn, OTHER = None):
    """Returns True if your next_move_fcn successfully guides the hunter_bot
    to the target_bot. This function is here to help you understand how we
    will grade your submission."""
    max_distance = 0.98 * target_bot.distance # 0.98 is an example. It will change.
    separation_tolerance = 0.02 * target_bot.distance # hunter must be within 0.02 step size to catch target
    caught = False
    ctr = 0
    #For Visualization
    import turtle
    window = turtle.Screen()
    window.bgcolor('white')
    chaser_robot = turtle.Turtle()
    chaser_robot.shape('arrow')
    chaser_robot.color('blue')
    chaser_robot.resizemode('user')
    chaser_robot.shapesize(0.3, 0.3, 0.3)
    broken_robot = turtle.Turtle()
    broken_robot.shape('turtle')
    broken_robot.color('green')
    broken_robot.resizemode('user')
    broken_robot.shapesize(0.3, 0.3, 0.3)
    size_multiplier = 15.0 #change size of animation
    chaser_robot.hideturtle()
    chaser_robot.penup()
    chaser_robot.goto(hunter_bot.x*size_multiplier, hunter_bot.y*size_multiplier-100)
    chaser_robot.showturtle()
    broken_robot.hideturtle()
    broken_robot.penup()
    broken_robot.goto(target_bot.x*size_multiplier, target_bot.y*size_multiplier-100)
    broken_robot.showturtle()
    measuredbroken_robot = turtle.Turtle()
    measuredbroken_robot.shape('circle')
    measuredbroken_robot.color('red')
    measuredbroken_robot.penup()
    measuredbroken_robot.resizemode('user')
    measuredbroken_robot.shapesize(0.1, 0.1, 0.1)
    broken_robot.pendown()
    chaser_robot.pendown()
    #End of Visualization
    # We will use your next_move_fcn until we catch the target or time expires.
    while not caught and ctr < 1000:
        # Check to see if the hunter has caught the target.
        hunter_position = (hunter_bot.x, hunter_bot.y)
        target_position = (target_bot.x, target_bot.y)
        separation = distance_between(hunter_position, target_position)
        if separation < separation_tolerance:
            print("You got it right! It took you ", ctr, " steps to catch the target.")
            caught = True

        # The target broadcasts its noisy measurement
        target_measurement = target_bot.sense()

        # This is where YOUR function will be called.
        turning, distance, OTHER = next_move_fcn(hunter_position, hunter_bot.heading, target_measurement, max_distance, OTHER)

        # Don't try to move faster than allowed!
        if distance > max_distance:
            distance = max_distance

        # We move the hunter according to your instructions
        hunter_bot.move(turning, distance)

        # The target continues its (nearly) circular motion.
        target_bot.move_in_circle()
        #Visualize it
        measuredbroken_robot.setheading(target_bot.heading*180/pi)
        measuredbroken_robot.goto(target_measurement[0]*size_multiplier, target_measurement[1]*size_multiplier-100)
        measuredbroken_robot.stamp()
        broken_robot.setheading(target_bot.heading*180/pi)
        broken_robot.goto(target_bot.x*size_multiplier, target_bot.y*size_multiplier-100)
        chaser_robot.setheading(hunter_bot.heading*180/pi)
        chaser_robot.goto(hunter_bot.x*size_multiplier, hunter_bot.y*size_multiplier-100)
        #End of visualization
        ctr += 1
        if ctr >= 1000:
            print("It took too many steps to catch the target.")
    return caught


target = robot(0.0, 10.0, 0.0, 2*pi / 30, 1.5)
# measurement_noise = .05*target.distance
# target.set_noise(0.0, 0.0, measurement_noise)

hunter = robot(-10.0, -10.0, 0.0)

# print(demo_grading(hunter, target, naive_next_move))
print(demo_grading2(hunter, target, next_move))