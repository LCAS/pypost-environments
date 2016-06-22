import numpy as np
from pypostEnvironments.dynamicalSystem.ContinuousTimeDynamicalSystem import ContinuousTimeDynamicalSystem
from pypostEnvironments.planarKinematics.PlanarForwardKinematics import PlanarForwardKinematics

class Pendulum(ContinuousTimeDynamicalSystem, PlanarForwardKinematics):

    def __init__(self, rootSampler):
        PlanarForwardKinematics.__init__(self, rootSampler.dataManager, 1)
        ContinuousTimeDynamicalSystem.__init__(self, rootSampler, 1)

        self.periodicRange = [-0.5 * np.pi, 1.5 * np.pi]
        self.maxTorque = 30
        self.noiseState = 0
        self.stateMinRange = np.asarray([-np.pi, -20])
        self.stateMaxRange = np.asarray([ np.pi,  20])
        self.actionMaxRange = np.asarray([500])

        self.linkProperty('periodicRange')
        self.linkProperty('maxTorque')
        self.linkProperty('noiseState')
        self.linkProperty('stateMinRange', 'pendulumStateMinRange')
        self.linkProperty('stateMaxRange', 'pendulumStateMaxRange')
        self.linkProperty('actionMaxRange', 'pendulumActionMaxRange')

        self.lengths = 0.5
        self.masses = 10
        self.inertias = self.masses * self.lengths**2 / 3
        self.g = 9.81
        self.sim_dt = 1e-4
        self.friction = 0.2

        self.dataManager.setRange('states', self.stateMinRange, self.stateMaxRange)
        self.dataManager.setRange('actions', - self.actionMaxRange, self.actionMaxRange)

        self.initObject()
        #Dummy
        self.addDataManipulationFunction(self.sampleAction, [], ['actions'])
        self.stepManager.addDataEntry('rewards', 1, -1, 1)
        self.addDataManipulationFunction(self.sampleReward, [], ['rewards'])
        self.addDataFunctionAlias('sampleReturn', 'sampleReward')



    def getExpectedNextStateContTime(self, dt, states, actions, *args):

        actions = np.maximum(-self.maxTorque, np.minimum(actions, self.maxTorque))
        nSteps = dt / self.sim_dt

        if nSteps != np.round(nSteps):
            # Todo make nice warning
            print('Warning from Pendulum: dt does not match up')
            nSteps = np.round(nSteps)

        c = self.g * self.lengths * self.masses / self.inertias
        for i in range(0, int(nSteps)):
            bla2 = states[:, 1:2] * self.friction
            bla3 = actions / self.inertias
            bla4 = c * np.sin(states[:, 0:1])
            bla1 = self.sim_dt * (bla4 + bla3 - bla2 )
            velNew = states[:, 1:2] + bla1
            states = np.concatenate((states[:, 0:1] + self.sim_dt * velNew, velNew), axis=1)
        return states

    # Below here are just dummy functions for testing

    def sampleReward(self, numElem):
        return np.zeros((numElem, 1))

    def sampleAction(self, numElem):
        return np.zeros([numElem, self.dimAction])