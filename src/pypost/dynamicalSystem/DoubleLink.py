import numpy as np
from pypost.dynamicalSystem.DynamicalSystem import DynamicalSystem

from pypost.dynamicalSystem.forwardModels import ForwardModel
from pypost.planarKinematics.PlanarForwardKinematics import PlanarForwardKinematics


class DoubleLink(DynamicalSystem, PlanarForwardKinematics):

    def __init__(self, dataManager):
        PlanarForwardKinematics.__init__(self, dataManager, 2)
        DynamicalSystem.__init__(self, dataManager, 2)

        self.lengths = np.asarray([1, 1])
        self.masses = np.asarray([1, 1])
        self.friction = np.asarray([0.025, 0.025])

        self.stateMinRange = np.asarray([-np.pi, -30, -np.pi, -30])
        self.stateMaxRange = np.asarray([ np.pi,  30,  np.pi,  30])
        self.maxTorque = np.asarray([10,  10])

        self.linkProperty('stateMinRange', 'stateMinRange')
        self.linkProperty('stateMaxRange', 'stateMaxRange')
        self.linkProperty('maxTorque', 'maxTorque')


        self.dataManager.setRange('states', self.stateMinRange, self.stateMaxRange)
        self.dataManager.setRange('actions', - self.maxTorque, self.maxTorque)

        self.inertias = self.masses * (self.lengths**2 + 0.0001) / 3.0
        self.g = 9.81
        self.sim_dt = 1e-4
        self.PDSetPoints = 0
        self.PDGains = 0

    def transitionFunction(self, states, actions):


        actions = np.maximum(-self.maxTorque, np.minimum(actions, self.maxTorque))
        actionNoise = actions + self.getControlNoise(states, actions)

        x_temp = ForwardModel.simulate_double_link(states, actionNoise, self.lengths, self.masses,
                                                        self.inertias, self.g, self.friction, self.dt, self.sim_dt)
        # always zeros, due to c implementation
        nextState = x_temp[:, :4]
        # can not return ffwdTorque here
        return nextState #. ffwdTorque

        # Todo Matlab references a c-file for linearized dynamics that does not exist in toolbox? port it when found
