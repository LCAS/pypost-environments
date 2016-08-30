import unittest

import numpy as np

from pypost.dynamicalSystem.forwardModels import ForwardModel


# IMPORTANT:
# Tests only verify correctness of output shapes, not actual values due to lack of reference model!
class Test(unittest.TestCase):

    def setUp(self):
        self.lengths = np.asarray([1, 1, 1, 1])
        self.masses = np.asarray([1, 1, 1, 1])
        self.friction = np.asarray([0, 0, 0, 0])
        self.inertias = self.masses * (self.lengths**2 + 0.0001) / 3.0
        self.g = 9.81

    def testDoubleLinkModel(self):
        states = np.zeros((10, 4))
        actions = np.ones((10, 2))

        next_states = ForwardModel.simulate_double_link(states, actions,
                                                             self.lengths[:2], self.masses[:2],
                                                             self.inertias[:2], self.g, self.friction[:2],
                                                        dt=1e-2, dst=1e-4)
        self.assertEqual(next_states.shape, (10, 6))

        next_states = ForwardModel.simulate_double_link(states, actions,
                                                        self.lengths, self.masses,
                                                        self.inertias, self.g, self.friction,
                                                        dt=1, dst=1e-4)
        self.assertEqual(next_states.shape, (10, 2))



    def testQuadLinkModel(self):
        states = np.zeros((10, 8))
        actions = np.ones((10, 4))

        next_states = ForwardModel.simulate_quad_link(states, actions,
                                                      self.lengths, self.masses,
                                                      self.inertias, self.g, self.friction,
                                                      dt=1e-2, dst=1e-4)
        self.assertEqual(next_states.shape, (10, 12))

        next_states = ForwardModel.simulate_quad_link(states, actions,
                                                      self.lengths, self.masses,
                                                      self.inertias, self.g, self.friction,
                                                      dt=1, dst=1e-4)
        self.assertEqual(next_states.shape, (10, 4))