""" Setup Module for pypost-environments.
"""

# Always prefer setuptools over distutils
from setuptools import setup, Extension

# To use a consistent encoding
from codecs import open
from os import path
import numpy

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# path to forward models
fmPath = "src/pypost/dynamicalSystem/forwardModels/"

# Install Dual Link Forward Model
_DoubleLinkForwardModel = Extension(fmPath + "_DoubleLinkForwardModel",
                                    [fmPath + "DoubleLinkForwardModel.i",
                                     fmPath + "DoubleLinkForwardModel.c"],
                                    include_dirs=[numpy_include])

# Install Quad Link Forward Model
_QuadLinkForwardModel = Extension(fmPath + "_QuadLinkForwardModel",
                                  [fmPath + "QuadLinkForwardModel.i",
                                   fmPath + "QuadLinkForwardModel.c"],
                                  include_dirs=[numpy_include],
                                  #extra_compile_args=['-fopenmp'],
                                  #extra_link_args=['-lgomp']
                                  )

setup(
    # TODO change credentials
    name='PyPoST Environments',
    version='0.0.1',
    author='Philipp Becker',
    author_email='philippbecker93@googlemail.com',
    description='Environments for PyPoST',
    long_description=long_description,
    url='http://www.ausy.tu-darmstadt.de',
    license='unknown',

    classifiers=[
        'Programming Language :: Python :: 3.5',
    ],

    #keywords='reinforcement learning',

    # pypost package is found in subdirectory src/
    package_dir={'': 'src'},

    # TODO use find_packages instead of manual listing.
    #packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    namespace_packages=['pypost'],

    packages=['pypost.dynamicalSystem',
              'pypost.dynamicalSystem.forwardModels',
              'pypost.planarKinematics',
              'pypost.preprocessor'],
    # TODO Fix error concerning pyyaml directory and uncomment
    # install_requires=['pyyaml', 'scipy']
    # TODO Convert to Python Wheels for security reasons

    ext_modules=[_DoubleLinkForwardModel,
                 _QuadLinkForwardModel],
)
