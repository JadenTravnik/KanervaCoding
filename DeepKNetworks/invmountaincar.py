'''
mountain car modified to have an extra penalty for non-zero action
'''

from pylab import random, sin
import math

def init():
    position = random()*.01 - .005
    return position, 0.0

def sample(S,A):
    position,velocity = S
    if A < -1 or A > 1:
        print 'Invalid action:', A
        raise StandardError
    R = 1
    velocity += 0.1*A - 0.0025*sin(1.5*position + math.pi/2.) - .0025
    if velocity < -0.07:
        velocity = -0.07
    elif velocity >= 0.07:
        velocity = 0.06999999
    position += velocity
    if abs(position) >= 1:
        return R,None

    return R,(position,velocity)
