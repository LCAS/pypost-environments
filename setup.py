""" Setup Module for pypost-environments.
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    # TODO change credentials
    name='PyPostEnvironments',
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
    package_dir={'':'src'},

    # TODO use find_packages instead of manual listing.
    #packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    packages=['pypostEnvironments',
              'pypostEnvironments.dynamicalSystem',
              'pypostEnvironments.dynamicalSystem.forwardModels',
              'pypostEnvironments.planarKinematics',
              'pypostEnvironments.preprocessor'],
    # TODO Fix error concerning pyyaml directory and uncomment
    # install_requires=['pyyaml', 'scipy']
    # TODO Convert to Python Wheels for security reasons
)