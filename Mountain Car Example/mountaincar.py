'''
mountain car modified to have an extra penalty for non-zero action
'''

from pylab import random, cos

def init():
    position = -0.6 + random()*0.2
    return position, 0.0

def sample(S,A):
    position,velocity = S
    if A < -1 or A > 1:
        print 'Invalid action:', A
        raise StandardError
    R = -1
    velocity += 0.001*A - 0.0025*cos(3*position)
    if velocity < -0.07:
        velocity = -0.07
    elif velocity >= 0.07:
        velocity = 0.06999999
    position += velocity
    if position >= 0.5:
        return R,None
    if position < -1.2:
        position = -1.2
        velocity = 0.0
    return R,(position,velocity)
