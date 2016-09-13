import numpy as np
from pypost.dynamicalSystem.ContinuousTimeDynamicalSystem import ContinuousTimeDynamicalSystem
from pypost.dynamicalSystem.forwardModels import ForwardModel
from pypost.planarKinematics.PlanarForwardKinematics import PlanarForwardKinematics


class QuadLink(ContinuousTimeDynamicalSystem, PlanarForwardKinematics):

    def __init__(self, dataManager):
        PlanarForwardKinematics.__init__(self, dataManager, 4)
        ContinuousTimeDynamicalSystem.__init__(self, dataManager, 4)

        # Todo use settings

        self.lengths = np.asarray([1, 1, 1, 1])
        self.masses = np.asarray([1, 1, 1, 1])
        self.friction = np.asarray([0, 0, 0, 0])
        self.minRangeState = np.asarray([-0.8, -50, -2.50, -50, -0.1, -50, -np.pi - 0.1, -50])
        self.maxRangeState = np.asarray([ 0.8,  50,  0.05,  50,  2  ,  50,  np.pi + 0.1,  50])
        self.minRangeAction = np.asarray([-10, -10, -10, -10])
        self.maxRangeAction = np.asarray([ 10,  10,  10,  10])
        self.inertias = self.masses * (self.lengths**2 + 0.0001) / 3.0
        self.g = 9.81
        self.sim_dt = 1e-4

        self.linkProperty('lengths', 'QuadLinkLengths')
        self.linkProperty('masses', 'QuadLinkMasses')
        self.linkProperty('friction', 'QuadLinkFrictions')
        self.linkProperty('g', 'QuadLinkGravity')
        self.linkProperty('sim_dt', 'QuadLinkSimDt')
        self.linkProperty('minRangeState', 'QuadLinkMinRangeState')
        self.linkProperty('maxRangeState', 'QuadLinkMaxRangeState')
        self.linkProperty('minRangeAction', 'QuadLinkMinRangeAction')
        self.linkProperty('maxRangeAction', 'QuadLinkMaxRangeAction')

        #todo check if works, pull up? sub data manager?
        self.dataManager.setRange('states', self.minRangeState, self.maxRangeState)
        self.dataManager.setRange('actions', self.minRangeAction, self.maxRangeAction)

        #self.initObject()

    def getExpectedNextStateContTime(self, states, actions, *args):

        x_temp = ForwardModel.simulate_quad_link(states, actions, self.lengths, self.masses, self.inertias,
                                                      self.g, self.friction, self.dt, self.sim_dt)
        ffwdTorque = x_temp[:, 8:]
        nextState = x_temp[:, :8]

        return nextState

    # Todo: Port Mex Implementation for linearized system and utilize it here