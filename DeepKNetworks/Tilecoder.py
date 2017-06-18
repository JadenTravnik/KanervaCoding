# Tilecoding for Mountain Car problem

import math 
numTilings = 8
numTiles = numTilings * 9 * 9
limitPos = 1.7       # 0.5 -(-1.2), i.e, (max - min) of Position
limitVel = 0.14      # 0.07 -(-0.07), i.e, (max - min) of Velocity
    
def tilecode(in1,in2,tileIndices):
    in1 += 1.2
    in2 += 0.07 
    for i in range (0, numTilings):
        x_off = i * (limitPos/8.0) / numTilings
        y_off = i * (limitVel/8.0) / numTilings        
        index1 = int(math.floor (8 * (in1 + x_off)/limitPos))
        index2 = int(math.floor (8 * (in2 + y_off)/limitVel))
        tileIndices[i] = int((81 * i) + (9 * index2) + index1)
    
    
def printTileCoderIndices(in1,in2):
    tileIndices = [-1]*numTilings
    tilecode(in1,in2,tileIndices)
    print 'Tile indices for input (',in1,',',in2,') are : ', tileIndices

'''
printTileCoderIndices(-1.2,-0.07)
printTileCoderIndices(-1.12,-0.07)
printTileCoderIndices(-1.10,-0.07)
printTileCoderIndices(0.5,0.07)
printTileCoderIndices(-1.2,0.07)
printTileCoderIndices(0.5,-0.07)
'''


