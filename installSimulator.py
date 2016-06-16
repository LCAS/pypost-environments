from setuptools import setup, Extension

# define the extension module
doublePendulumForwardModel = Extension('doublePendulumForwardModel',
                                       sources=['src/pypostEnvironments/dynamicalSystem/DoublePendulumForwardModel.c'])

# run the setup
setup(name='doublePendulumForwardModel',
      version='1.0',
      description='test',
      ext_modules=[doublePendulumForwardModel])