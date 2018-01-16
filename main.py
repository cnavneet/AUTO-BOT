from robot import *
from math import *
from matrix import *
import random

def kFilt(x, P, measurement, u): # Basic Kalman Filter
    # prediction
    x = (F * x) + u
    P = F * P * F.transpose()
    
    # measurement update
    Z = matrix([measurement])
    y = Z.transpose() - (H * x)
    S = H * P * H.transpose() + R
    K = P * H.transpose() * S.inverse()
    x = x + (K * y)
    P = (I - (K * H)) * P
    
    return x, P

def pFilt(p, measurements, N=500): # I know it's tempting, but don't change N!
    # --------
    # Make particles
    if p == None: # Particles not already made-no guess provided
        p = []
        for i in range(N):
            r = matrix([[random.uniform(-50.0,50.0)],[random.uniform(-50.0,50.0)],[random.uniform(0,50)],[random.uniform(0,2*pi)],[random.uniform(-pi,pi)]])
            p.append(r)
    elif len(p) == 5: # Particles not already made-First guess provided
        p2 = []
        for i in range(N):
            r = matrix([[random.uniform(p[0]-0.75,p[0]+0.75)],[random.uniform(p[1]-0.75,p[1]+0.75)],[random.uniform(max(p[2]-0.75,0),p[2]+0.75)],[random.uniform((p[3]-pi/1.5)%(2*pi),(p[3]+pi/1.5)%(2*pi))],[random.uniform(p[4]-pi/12,p[4]+pi/12)]])
            p2.append(r)
        p = p2

    N = len(p)

    # --------
    # Update particles

    # motion update (prediction)
    p2 = [F*i+u for i in p]

    # measurement update
    w = [] # Weight of each
    diff = []
    for i in range(N):
        position = get_position(p[i])
        diff.append(sqrt((position[0]-measurements[0])^2+(position[1]-measurements[1]^2))) # Distance between the predicted point and the actual point
    maxDiff = max(diff)
    w = [maxDiff-diff[i] for i in range(N)]

    # resampling - Using Sampling Wheel
    p3 = []
    index = int(random.random() * N)
    beta = 0.0
    mw = max(w)
    for i in range(N):
        beta += random.random() * 2.0 * mw
        while beta > w[index]:
            beta -= w[index]
            index = (index + 1) % N
        p3.append(p[index])
    p = p3
    
    return p

def get_position(x, mode="current"):
    center = (x.value[0][0], x.value[1][0])
    r = x.value[2][0]
    x.value[3][0] = x.value[3][0]%(2*pi)
    theta = x.value[3][0]
    if mode=="next":
        theta_dot = x.value[4][0]
    else:
        theta_dot = 0

    new_x = r*cos(theta+theta_dot)+center[0]
    new_y = r*sin(theta+theta_dot)+center[1]
    return (new_x, new_y)

def estimate_next_pos(measurement, OTHER = None):
    """Estimate the next (x, y) position of the wandering Traxbot
        based on noisy (x, y) measurements."""
    # Kalman Filter for r, angle, rotVel
    # X = [center_x, center_y, r, angle, rotVel]
    xy_estimate = (0.0, 0.0)
    x = []
    mode = "c"
    
    # MEASUREMENT UPDATE
    if mode=="k":
        if OTHER==None: # First Timestep- create OTHER
            OTHER = {'z':[measurement], 'p':P, 'x': matrix([[0.0], [0.0], [1.0], [0.0], [0.1]])};
        elif len(OTHER['z'])==1: # Second Timestep, wait
            OTHER['z'].append(measurement)
        else: # 3rd+ Timestep, Calculate
            # Find the radius, center of the circle and the rotational velocity
            # This is only done on timestep 3 (first moment this is possible)
            OTHER['z'].append(measurement)
            if len(OTHER['z']) > 7:
                OTHER['z'].pop(0)
            mid = int(len(OTHER['z'])/2.0)

            # Step 1: Find Bisection Points
            bisect = [[0.5 * (OTHER['z'][mid][i] + OTHER['z'][0][i]) for i in range(2)]]
            bisect.append([0.5 * (OTHER['z'][-1][i] + OTHER['z'][mid][i]) for i in range(2)])

            # Step 2: Solve for slope of two line segments formed by robot motion
            slope = [(OTHER['z'][mid][1] - OTHER['z'][0][1])/(OTHER['z'][mid][0] - OTHER['z'][0][0])]
            slope.append((OTHER['z'][-1][1] - OTHER['z'][mid][1])/(OTHER['z'][-1][0] - OTHER['z'][mid][0]))

            # Step 3: For each segment find slope of perpendicular line
            perpSlope = [-1.0/i for i in slope]

            # Step 4: Find the y intecept of a line perpendicular to each of the line
            #  segments from step 2 at their bisection points found in step 1
            yInt = [ bisect[i][1] - (perpSlope[i]*bisect[i][0]) for i in range(2) ]

            # Step 5: Find the intersection of these two lines, this is the center of the circle
            x = [ (yInt[0] - yInt[1]) / (perpSlope[1] - perpSlope[0]) ] # x
            x.append( (x[0] * perpSlope[0]) + yInt[0] ) # y

            # Step 6: Solve for the radius
            x.append(distance_between( tuple(x), measurement ) )

            # Step 7: Find the angle of the vector from the center to
            #   each point (from the x axis) (rads)
            x.append([atan2(OTHER['z'][i][1]-x[1],OTHER['z'][i][0]\
                                   -x[0]) for i in range(len(OTHER['z']))])
            # Step 8: Find the rotational velocity (rads/s)
            motion = [(x[3][i+1] - x[3][i] + pi)%(2*pi)-pi for i in range(len(x[3])-1)]
            x.append(sum(motion)/len(motion))
            x[3] = x[3][-1]

            # PREDICT
            OTHER['x'], OTHER['p'] = kFilt(OTHER['x'], OTHER['p'], x, u) # Kalman Filter

    elif mode == "p":
        if OTHER==None: # First Timestep- create OTHER
            OTHER = {"p":None}
        OTHER['p'] = pFilt(OTHER['p'],measurement,5000)
        OTHER['x'] = OTHER['p'][0]

    elif mode == "c":
        if OTHER==None: # First Timestep- create OTHER
            OTHER = {'z':[measurement], 'p':P, 'x': matrix([[0.0], [0.0], [1.0], [0.0], [0.1]])};
        elif len(OTHER['z'])<4: # 2nd,3rd timestep
            OTHER['z'].append(measurement)
        elif len(OTHER['z'])<12: # 7th Timestep, Calculate
            # Find the radius, center of the circle and the rotational velocity
            # This is only done on timestep 3 (first moment this is possible)
            OTHER['z'].append(measurement)
            mid = int(len(OTHER['z'])/2.0)

            # Step 1: Find Bisection Points
            bisect = [[0.5 * (OTHER['z'][mid][i] + OTHER['z'][0][i]) for i in range(2)]]
            bisect.append([0.5 * (OTHER['z'][-1][i] + OTHER['z'][mid][i]) for i in range(2)])
        
            # Step 2: Solve for slope of two line segments formed by robot motion
            slope = [(OTHER['z'][mid][1] - OTHER['z'][0][1])/(OTHER['z'][mid][0] - OTHER['z'][0][0])]
            slope.append((OTHER['z'][-1][1] - OTHER['z'][mid][1])/(OTHER['z'][-1][0] - OTHER['z'][mid][0]))
            
            # Step 3: For each segment find slope of perpendicular line
            perpSlope = [-1.0/i for i in slope]
            
            # Step 4: Find the y intecept of a line perpendicular to each of the line
            #  segments from step 2 at their bisection points found in step 1
            yInt = [ bisect[i][1] - (perpSlope[i]*bisect[i][0]) for i in range(2) ]
            
            # Step 5: Find the intersection of these two lines, this is the center of the circle
            x = [ (yInt[0] - yInt[1]) / (perpSlope[1] - perpSlope[0]) ] # x
            x.append( (x[0] * perpSlope[0]) + yInt[0] ) # y
            
            # Step 6: Solve for the radius
            x.append(distance_between( tuple(x), measurement ) )
            
            # Step 7: Find the angle of the vector from the center to
            #   each point (from the x axis) (rads)
            x.append([atan2(OTHER['z'][i][1]-x[1],OTHER['z'][i][0]\
                            -x[0]) for i in range(len(OTHER['z']))])
            # Step 8: Find the rotational velocity (rads/s)
            motion = [(x[3][i+1] - x[3][i] + pi)%(2*pi)-pi for i in range(len(x[3])-1)]
            x.append(sum(motion)/len(motion))
            x[3] = x[3][-1]
            
            # PREDICT
            OTHER['x'], OTHER['p'] = kFilt(OTHER['x'], OTHER['p'], x, u) # Kalman Filter
            if len(OTHER['z'])==12:
                x = [OTHER['x'].value[i][0] for i in range(5)]
                OTHER['p'] = pFilt(x, measurement, 30000)
                OTHER['x'] = OTHER['p'][0]
        else:
            OTHER['p'] = pFilt(OTHER['p'], measurement)
            OTHER['x'] = OTHER['p'][0]

    xy_estimate = get_position(OTHER['x'], mode="next")
    print xy_estimate
    return xy_estimate, OTHER

# A helper function you may find useful.
def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# This is here to give you a sense for how we will be running and grading
# your code. Note that the OTHER variable allows you to store any
# information that you want.
def demo_grading(estimate_next_pos_fcn, target_bot, OTHER = None):
    localized = False
    distance_tolerance = 0.01 * target_bot.distance
    ctr = 0
    # if you haven't localized the target bot, make a guess about the next
    # position, then we move the bot and compare your guess to the true
    # next position. When you are close enough, we stop checking.
    while not localized and ctr <= 1000:
        ctr += 1
        print ctr
        measurement = target_bot.sense()
        position_guess, OTHER = estimate_next_pos_fcn(measurement, OTHER)
        target_bot.move_in_circle()
        true_position = (target_bot.x, target_bot.y)
        print true_position
        correct[3] = ((3.0*pi/2.0+target_bot.heading)-correct[4]/2.0)%(2.0*pi)
        #raw_input()
        error = distance_between(position_guess, true_position)
        if error <= distance_tolerance:
            print "You got it right! It took you ", ctr, " steps to localize."
            localized = True
        if ctr == 1000:
            print "Sorry, it took you too many steps to localize the target."
    return localized

# This is a demo for what a strategy could look like. This one isn't very good.
def naive_next_pos(measurement, OTHER = None):
    """This strategy records the first reported position of the target and
        assumes that eventually the target bot will eventually return to that
        position, so it always guesses that the first position will be the next."""
    if not OTHER: # this is the first measurement
        OTHER = measurement
    xy_estimate = OTHER
    return xy_estimate, OTHER

# This is how we create a target bot. Check the robot.py file to understand
# How the robot class behaves.
test_target = robot(2.1, 4.3, 0.5, 2*pi / 34.0, 1.5)
measurement_noise = 0.01 * test_target.distance
test_target.set_noise(0.0, 0.0, measurement_noise)
correct = [ 2.1-8.10532*cos((1.5*pi+0.5)-(pi/34.0)), 4.3-8.10532*sin((1.5*pi+0.5)-(pi/34.0)), 8.10532, (1.5*pi+0.5)-(pi/34.0), 2*pi / 34.0 ]

P = matrix([[1000.0, 0.0, 0.0, 0.0, 0.0],\
            [0.0, 1000.0, 0.0, 0.0, 0.0],\
            [0.0, 0.0, 1000.0, 0.0, 0.0],\
            [0.0, 0.0, 0.0, 1000.0, 0.0],\
            [0.0, 0.0, 0.0, 0.0, 1000.0]])# initial uncertainty
F = matrix([[1.0, 0.0, 0.0, 0.0, 0.0],\
            [0.0, 1.0, 0.0, 0.0, 0.0],\
            [0.0, 0.0, 1.0, 0.0, 0.0],\
            [0.0, 0.0, 0.0, 1.0, 1.0],\
            [0.0, 0.0, 0.0, 0.0, 1.0]]) # next state function
H = matrix([[1.0, 0.0, 0.0, 0.0, 0.0],\
            [0.0, 1.0, 0.0, 0.0, 0.0],\
            [0.0, 0.0, 1.0, 0.0, 0.0],\
            [0.0, 0.0, 0.0, 1.0, 0.0],\
            [0.0, 0.0, 0.0, 0.0, 1.0]])# measurement function
R = matrix([[5.0, 0.0, 0.0, 0.0, 0.0],\
            [0.0, 15.0, 0.0, 0.0, 0.0],\
            [0.0, 0.0, 15.0, 0.0, 0.0],\
            [0.0, 0.0, 0.0, 50.0, 0.0],\
            [0.0, 0.0, 0.0, 0.0, 4e-3]])# measurement uncertainty
I = matrix([[1.0, 0.0, 0.0, 0.0, 0.0],\
            [0.0, 1.0, 0.0, 0.0, 0.0],\
            [0.0, 0.0, 1.0, 0.0, 0.0],\
            [0.0, 0.0, 0.0, 1.0, 0.0],\
            [0.0, 0.0, 0.0, 0.0, 1.0]])# identity matrix
u = matrix([[0.0] for i in range(5)])


result = demo_grading(estimate_next_pos, test_target)