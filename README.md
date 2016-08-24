
# Kanerva Coding
Kanerva Coding (KC) is a way of representing values of a vector of continous variables.

The KC software provides 2 public routines:


## void Initialize(int numPrototypes, int dimension, String distanceMeasure)

The distanceMeasure parameter is used to indicate how the adjacency is calculated between prototypes and state observations.

The prototypes are positioned randomly in the space with no prevention of overlapping prototypes.
All prototypes lay within an n-dimensional cube denoted by the dimension parameter.
The user must normalize their data when requesting GetFeatures


### adaptive
The prototypes will be adaptively deleted and reinitialized around the statespace by using methods described in this [paper](https://www.semanticscholar.org/paper/Reinforcement-learning-with-adaptive-Kanerva-Allen-Fritzsche/918088cf686bcf75deaa768b1baea250c4f52ac5/pdf "Reinforcement Learning with Adaptive Kanerva Coding for Xpilot Game AI"). 


## int GetFeatures(float data[])

Uses the observations defined in data[] to compute the binary features indicating adjacentcy to prototypes.


