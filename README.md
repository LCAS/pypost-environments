# PyPoST Environments

Adds Simulators for dynamical Systems to the PyPoST

## Dynamical Systems

Currently implemented

1. Pendulum

2. Double Link

3. Quad Link

## Preprocessor 

The preprocessor interface allows 

Currently implemented preprocessors:

PlanarKinematicsImagePreprocessor - renders 2D images of planar kinematics systems like pendulum, double link and quad link

## Installation

Both, the Double and the Quad Link, are simulated using C code for performance reasons. The interfacing is done using SWIG, if SWIG is installed, the C-Modules should be installed automatically.