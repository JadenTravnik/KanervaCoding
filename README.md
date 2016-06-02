
# Kanerva Coding
Kanerva Coding (KC) is a way of representing values of a vector of continous variables.

The KC software provides 2 public routines:


## void Initialize(int numPrototypes, float lowerBounds[], float upperBounds[], String spacingType, String distanceMeasure)

Uses the upper and lower bounds to initialize prototypes within the n-dimensional statespace.
The lowerBounds and upperBounds arrays should be equal length where (lowerBounds[i], upperBounds[i]) define the limits for the ith observation given in GetFeatures.

As the approach of Kanerva Coding is highly sensitive to the number and distribution of the set of prototypes chosen, several differet options are available:

The distanceMeasure parameter is used to indicate how the adjacency is calculated between prototypes and state observations.

The spacingType parameter indicates which type of positioning is used:

### random
The prototypes are positioned randomly in the space with no prevention of overlapping prototypes.

### uniform
The prototypes are positioned uniformly randomly using the distance measure indicated. It ensures that all prototypes are at least

### adaptive
The prototypes will be adaptively deleted and reinitialized around the statespace by using methods described in this [paper](https://www.semanticscholar.org/paper/Reinforcement-learning-with-adaptive-Kanerva-Allen-Fritzsche/918088cf686bcf75deaa768b1baea250c4f52ac5/pdf "Reinforcement Learning with Adaptive Kanerva Coding for Xpilot Game AI"). 


## int GetFeatures(float data[])

Uses the observations defined in data[] to compute the binary features indicating adjacentcy to prototypes.


