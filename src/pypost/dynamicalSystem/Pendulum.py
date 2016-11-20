import numpy as np

from pypost.dynamicalSystem import DynamicalSystem
from pypost.planarKinematics.PlanarForwardKinematics import PlanarForwardKinematics


class Pendulum(DynamicalSystem, PlanarForwardKinematics):

    def __init__(self, dataManager):
        PlanarForwardKinematics.__init__(self, dataManager, 1)
        DynamicalSystem.__init__(self, dataManager, 1)

        self.maxTorque = 30
        self.noiseState = 0
        self.stateMinRange = np.asarray([-np.pi, -20])
        self.stateMaxRange = np.asarray([ np.pi,  20])

        self.linkProperty('stateMinRange', 'stateMinRange')
        self.linkProperty('stateMaxRange', 'stateMaxRange')
        self.linkProperty('maxTorque', 'maxTorque')

        self.lengths = 0.5
        self.masses = 10
        self.inertias = self.masses * self.lengths**2 / 3
        self.g = 9.81
        self.sim_dt = 1e-4
        self.friction = 0.2

        self.dataManager.setRange('states', self.stateMinRange, self.stateMaxRange)
        self.dataManager.setRange('actions', - self.maxTorque, self.maxTorque)

       # self.initObject()

    def transitionFunction(self, states, actions):

        actions = np.maximum(-self.maxTorque, np.minimum(actions, self.maxTorque))
        actionNoise = actions + self.getControlNoise(states, actions)
        nSteps = self.dt / self.sim_dt

        if nSteps != np.round(nSteps):
            print('Warning from Pendulum: dt does not match up')
            nSteps = np.round(nSteps)

        c = self.g * self.lengths * self.masses / self.inertias
        for i in range(0, int(nSteps)):
            velNew = states[:, 1:2] + self.sim_dt * (c * np.sin(states[:, 0:1])
                                                     + actionNoise / self.inertias
                                                     - states[:, 1:2] * self.friction )
            states = np.concatenate((states[:, 0:1] + self.sim_dt * velNew, velNew), axis=1)
        return states

