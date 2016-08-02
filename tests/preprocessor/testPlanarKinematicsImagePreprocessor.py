import numpy as np
import unittest
from pypostEnvironments.preprocessor.PlanarKinematicsImagePreprocessor import Renderer

class TestRenderer(unittest.TestCase):

    def setUp(self):
        self.r = Renderer(48, 2)

        self.state1 = np.asarray([0, 1, 0, 1])
        self.onlyPosState1 = np.asarray([0, 0])


    def test_range(self):
        img = self.r.generateImageFromState(self.state1)
        self.assertLessEqual(np.max(img), 1)
        self.assertGreaterEqual(np.min(img), 0)

    def test_generation(self):
        # times two and up side down during generation
        target_points = np.asarray([[48, 48], [48, 72], [48, 96]])

        points = self.r._generatePointsFromAngles(self.onlyPosState1)
        self.assertTrue((points == target_points).all())
