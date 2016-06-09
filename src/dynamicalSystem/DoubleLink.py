from src.dynamicalSystem.ContinuousTimeDynamicalSystem import ContinuousTimeDynamicalSystem
from src.planarKinematics import PlanarForwardKinematics
import numpy as np


class DoubleLink(ContinuousTimeDynamicalSystem, PlanarForwardKinematics):

    def __init__(self, rootSampler):
        super(ContinuousTimeDynamicalSystem).__init__(rootSampler, 1)
        super(PlanarForwardKinematics).__init__(rootSampler, 1)

        self.lengths = np.asarray([1, 1])
        self.masses = np.asarray([1, 1])
        self.friction = np.asarray([0.025, 0.025])

        self.dataManager.setRange('states', np.asarray([[-np.pi, -30, -np.pi, -30],
                                                        [ np.pi,  30,  np.pi,  30]]))
        self.dataManager.setRange('actions', np.asarray([[-10, -10],
                                                         [ 10,  10]]))

        self.inertias = self.masses * (self.lengths**2 + 0.0001) / 3.0
        self.g = 9.81
        self.sim_dt = 1e-4
        self.PDSetPoints = 0
        self.PDGains = 0
        self.initObject()

    def getExpectedNextStateContTime(self, dt, states, actions, *args):

        nextState = np.zeros(np.shape(states))
        ffwdTorque = np.zeros(np.shape(states)[0], 2)

        # clip actions
        minRange = self.dataManager.getMinRange('actions')
        maxRange = self.dataManager.getMaxRange('actions')
        action = np.maximum(minRange, np.minimum(actions, maxRange))

        for i in range(0, np.shape(states)[0]):
            #Todo try to reuse c forward model or re implement directly in python
            #  x_temp =
            return
