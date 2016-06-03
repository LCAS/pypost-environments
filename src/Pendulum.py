from src.TransitionFunction import TransitionFunction
from pypost.data.DataManager import DataManager
import numpy as np

class Pendulum(TransitionFunction):
    def __init__(self, rootSampler):
        # State dim = 2, Action dim = 1
        super(Pendulum, self).__init__(rootSampler, 2, 1)

        self.numContext = 2
        self.initialState = np.asarray([0, 0])
        self.maxVelo = 8
        self.maxTorque = 2
        self.dt = 0.05

        self.g = 10
        self.m = 1
        self.l = 1

        self.episodeManager = rootSampler.dataManager
        self.stepManager = self.episodeManager.subDataManager

        self.episodeManager.addDataEntry('contexts', self.numContext,
                                         - np.ones(self.numContext), np.ones(self.numContext))

        self.stepManager.setRange('actions', np.asarray([- self.maxTorque]), np.asarray([self.maxTorque]))
        min_state = np.asarray([float('-inf'), - self.maxVelo])
        max_state = np.asarray([float('inf'), self.maxVelo])
        self.stepManager.setRange('states', min_state, max_state)
        self.stepManager.setRange('nextStates', min_state, max_state)

        self.stepManager.addDataEntry('rewards', 1)

        self.addDataManipulationFunction(self.sampleAction, [], ['actions'])
        self.addDataFunctionAlias('sampleParameter', 'sampleAction')
        self.addDataManipulationFunction(self.sampleReward, [], ['rewards'])
        self.addDataManipulationFunction(self.sampleContext, [], ['contexts'])
        self.addDataManipulationFunction(self.sampleInitialPosition, ['contexts'], ['states'])

    def transitionFunction(self, state, action):
        pos = state[:, 0:1]
        vel = state[:, 1:2]

        action = np.clip(action, -self.maxTorque, self.maxTorque)
        # a = u(t)/ml² - mgl sin(pos)
        sin_term = np.sin(pos + np.pi)
        a = -3 * self.g / (2 * self.l) * sin_term
        b = 3. / (self.m * self.l ** 2) * action
        d_vel = ( a+  b) * self.dt
        new_vel = vel + d_vel
        new_pos = pos + new_vel * self.dt
        new_vel = np.clip(new_vel, - self.maxVelo, self.maxVelo)
        return np.concatenate((new_pos, new_vel), 1)

    def sampleContext(self, num_elements):
        return np.tile(self.initialState, (num_elements, 1))

    def sampleInitialPosition(self, context):
        return np.reshape(context[:, 0], 2)

    def sampleAction(self, num_elements):
        return np.random.uniform(-self.maxTorque, self.maxTorque, (num_elements, 1))

    # Todo: implement meaningful cost functions
    def sampleReward(self, num_elements):
        return np.zeros((num_elements, 1))

    def _pendulumDynamics(self, t, y, action):
        pos = y[:, 0]
        vel = y[:, 1]
        d_pos = vel
        d_vel = action / (self.m * self.l**2) - self.m * self.g * self.l * np.sin(pos)
        return np.concatenate((d_pos, d_vel), 1)
# pendulum = Pendulum()


# states = np.zeros((1000, 2))
# actions = np.zeros((1000, 1))

# for i in range(0, 999):
#    actions[i] = pendulum.sampleAction()
#    next_state = pendulum.transitionFunction(states[i], actions[i])
#    states[i + 1] = next_state

# print('bla')
