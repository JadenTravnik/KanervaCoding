
# Kanerva Coding
Kanerva Coding (KC) is a way of representing values of a vector of continous variables (AKA Function Approximation)

The current repo is split into 2 main folders, one readily available script (kanerva.py) and an example of it's usage.

## Mountain Car Example

This folder has all of scripts from the Representation Heuristics Folder and allows for a quick view of the different functionality of each heuristic.

run with : python main.py

## Changing Representations
So far there are two methods from literature that have been implemented. Both of them involve manipulating the prototypes to adapt to the task being learned.
The first, refered to as [XGame](https://www.semanticscholar.org/paper/Reinforcement-learning-with-adaptive-Kanerva-Allen-Fritzsche/918088cf686bcf75deaa768b1baea250c4f52ac5/pdf "Reinforcement Learning with Adaptive Kanerva Coding for Xpilot Game AI") within the documentation, involves moving the prototypes to lay along the trajectory of the agent. This clearly has its faults, such as moving the prototypes before the state space has been thoroughly explored but has been used effectively to represent the most commonly visited states.

The second method, refered to as [Case Studies](https://www.aaai.org/Papers/Workshops/2004/WS-04-08/WS04-08-015.pdf "Sparse Distributed Memories in Reinforcement Learning: Case Studies") within the documentation, invovles reallocation of the "memory resources and has been shown to be robust in the context of RL and produces relatively compact representations of the action-value functions". In other words, it attempts to minimize the number of prototypes necessary to represent the state.

## Representation Heuristics 
This folder takes the different options found in kanerva.py in the topmost folder and breaks them out into different classes.

## Hierarchical KCoding
This is a very experimental approach to Representation and contains a branch of thinking that has no solid basis in literature.
I had the idea of constructing layers of sparse distributed prototypes, akin to deep learning but have yet to ground the ideas in literature so it is very experimental and untested.
As such it has minimal documentation.



