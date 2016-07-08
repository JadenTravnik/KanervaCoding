# Tilecoding for Mountain Car problem

import math 

class TileCoder1D:
    def __init__(self, _numTilings, _numTilesPerTiling, _maxState):
        self.numTilings = _numTilings
        self.numTilesPerTiling = _numTilesPerTiling
        self.numTiles = _numTilings * _numTilesPerTiling
        self.maxState = _maxState
        self.offset = (_maxState/float(_numTilings)) / _numTilings
        self.w = [0]*self.numTiles

    def tilecode(self, in1,tileIndices):
        for i in range (0, self.numTilings):
            x_off = i * self.offset    
            index1 = int(math.floor (self.numTilings * (in1 + x_off)/self.maxState))
            tileIndices[i] = int((self.numTilesPerTiling * i) + index1)


# returns the Q value for the state action pair
    def getV(self, state):
        tileIndices = [0] * self.numTilings
        self.tilecode(state, tileIndices)
        thetaSum = 0
        for index in tileIndices:
            thetaSum += self.w[index]

        return thetaSum


    # updates the theta values of the prototypes
    def updateWeights(self, state, delta, alpha):
        tileIndices = [0] * self.numTilings
        self.tilecode(state, tileIndices)

        for index in tileIndices:
            self.w[index] += alpha*delta