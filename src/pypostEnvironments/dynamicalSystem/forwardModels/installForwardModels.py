from distutils.core import Extension, setup

import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


# Install Dual Link Forward Model
_DoubleLinkForwardModel = Extension("_DoubleLinkForwardModel",
                               ["DoubleLinkForwardModel.i","DoubleLinkForwardModel.c"],
                               include_dirs = [numpy_include],
                               )

setup(  name        = "DoubleLinkForwardModel",
        description = "Simulates forward dynamics of Dual Link",
        author      = "Philipp",
        version     = "1.0",
        ext_modules = [_DoubleLinkForwardModel]
        )


# Install Quad Link Forward Model
_QuadLinkForwardModel = Extension("_QuadLinkForwardModel",
                               ["QuadLinkForwardModel.i","QuadLinkForwardModel.c"],
                               include_dirs = [numpy_include],
                               )

setup(  name        = "QuadLinkForwardModel",
        description = "Simulates forward dynamics of Quad Link",
        author      = "Philipp",
        version     = "1.0",
        ext_modules = [_QuadLinkForwardModel]
        )