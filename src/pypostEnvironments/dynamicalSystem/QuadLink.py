from pypostEnvironments.dynamicalSystem.ContinuousTimeDynamicalSystem import ContinuousTimeDynamicalSystem
from pypostEnvironments.planarKinematics.PlanarForwardKinematics import PlanarForwardKinematics
import pypostEnvironments.dynamicalSystem.ForwardSimWrapper as Simulator
import numpy as np

class QuadLink(ContinuousTimeDynamicalSystem, PlanarForwardKinematics):

    def __init__(self, root_sampler):
        PlanarForwardKinematics.__init__(self, root_sampler.dataManager, 4)
        ContinuousTimeDynamicalSystem.__init__(self, root_sampler, 4)

        self.lengths = np.asarray([1, 1, 1, 1])
        self.masses = np.asarray([1, 1, 1, 1])
        self.friction = np.asarray([0, 0, 0, 0])

        self.dataManager.setRange('states',
                                  np.asarray([-0.8, -50, -2.50, -50, -0.1, -50, -np.pi - 0.1, 50]),
                                  np.asarray([ 0.8,  50,  0.05,  50,  2  ,  50,  np.pi + 0.1, 50]))

        self.dataManager.setRange('actions',
                                  np.asarray([-10, -10, -10, -10]),
                                  np.asarray([ 10,  10,  10,  10]))

        self.inertias = self.masses * (self.lengths**2 + 0.0001) / 3.0
        self.g = 9.81
        self.sim_dt = 1e-4

        self.initObject()

        #Dummy
        self.addDataManipulationFunction(self.sampleAction, [], ['actions'])
        self.stepManager.addDataEntry('rewards', 1, -1, 1)
        self.addDataManipulationFunction(self.sampleReward, [], ['rewards'])
        self.addDataFunctionAlias('sampleReturn', 'sampleReward')
        self.maxTorque = 30

    def getExpectedNextStateContTime(self, dt, states, actions, *args):

        nextState = np.zeros(np.shape(states))
        ffwdTorque = np.zeros((np.shape(states)[0], 4))

        # clipping?

        for i in range(0, np.shape(states)[0]):
            x_temp = Simulator.simulate_quad_pendulum(states[i, :], actions[i, :], self.lengths, self.masses, self.inertias,
                                                      self.g, self.friction, self.dt, self.sim_dt)
            ffwdTorque[i, :] = x_temp[8:]
            nextState[i, :] = x_temp[:8]

        return nextState

  #  def getLinearizedContinuousTimeDynamics(self, states, actions, *args):
    def sampleReward(self, numElem):
        return np.zeros((numElem, 1))

    def sampleAction(self, numElem):
        action = np.random.uniform(- self.maxTorque, self.maxTorque, [numElem, self.dimAction])
        return action
