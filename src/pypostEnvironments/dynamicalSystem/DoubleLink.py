import numpy as np
from pypostEnvironments.dynamicalSystem.ContinuousTimeDynamicalSystem import ContinuousTimeDynamicalSystem
from pypostEnvironments.planarKinematics.PlanarForwardKinematics import PlanarForwardKinematics
import pypostEnvironments.dynamicalSystem.forwardModels.ForwardModelWrapper as Simulator


class DoubleLink(ContinuousTimeDynamicalSystem, PlanarForwardKinematics):

    def __init__(self, rootSampler):
        PlanarForwardKinematics.__init__(self, rootSampler.dataManager, 2)
        ContinuousTimeDynamicalSystem.__init__(self, rootSampler, 2)

        self.lengths = np.asarray([1, 1])
        self.masses = np.asarray([1, 1])
        self.friction = np.asarray([0.025, 0.025])

        self.maxTorque = 30
        self.noiseState = 0
        self.stateMinRange = np.asarray([-np.pi, -30, -np.pi, -30])
        self.stateMaxRange = np.asarray([ np.pi,  30,  np.pi,  30])
        self.actionMaxRange = np.asarray([10,  10])

        self.linkProperty('maxTorque')
        self.linkProperty('noiseState')
        self.linkProperty('stateMinRange', 'pendulumStateMinRange')
        self.linkProperty('stateMaxRange', 'pendulumStateMaxRange')
        self.linkProperty('actionMaxRange', 'pendulumActionMaxRange')


        self.dataManager.setRange('states', self.stateMinRange, self.stateMaxRange)
        self.dataManager.setRange('actions', - self.actionMaxRange, self.actionMaxRange)

        self.inertias = self.masses * (self.lengths**2 + 0.0001) / 3.0
        self.g = 9.81
        self.sim_dt = 1e-4
        self.PDSetPoints = 0
        self.PDGains = 0
        self.initObject()

    def getExpectedNextStateContTime(self, dt, states, actions, *args):

        nextState = np.zeros(np.shape(states))
        ffwdTorque = np.zeros((len(states), 2))

        minRange = self.dataManager.getMinRange('actions')
        maxRange = self.dataManager.getMaxRange('actions')
        action = np.maximum(minRange, np.minimum(actions, maxRange))

        x_temp = Simulator.simulate_double_link(states, action, self.lengths, self.masses,
                                                        self.inertias, self.g, self.friction, self.dt, self.sim_dt)
        # always zeros, due to c implementation
        ffwdTorque = x_temp[:, 4:]
        nextState = x_temp[:, :4]
        # can not return ffwdTorque here
        return nextState #. ffwdTorque

        # Todo Matlab references a c-file for linearized dynamics that does not exist in toolbox? port it when found
