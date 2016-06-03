from src.TransitionFunction import TransitionFunction
from pypost.data.DataManager import DataManager
import numpy as np

class Pendulum():
    def __init__(self):
        # Todo make parameters setable (via dict?)
        self.initialState = np.asarray([0, 0])
        self.maxVelo = 8
        self.maxTorque = 2
        self.dt = 0.05
        self.m = 1
        self.l = 1
        self.g = 10

    def transitionFunction(self, state, action):
        pos = state[0]
        vel = state[1]

        action = np.clip(action, -self.maxTorque, self.maxTorque)
        # magic formula, will be replaced by "ode45"-like solver
        new_vel = vel + (-3 * self.g / (2 * self.l) * np.sin(pos + np.pi) + 3. / (
            self.m * self.l ** 2) * action) * self.dt
        new_pos = pos + new_vel * self.dt
        new_vel = np.clip(new_vel, - self.maxVelo, self.maxVelo)
        return np.reshape(np.asarray([new_pos, new_vel]), (2))

    def sampleAction(self, shape):
        return np.random.uniform(-self.maxTorque, self.maxTorque, shape)
