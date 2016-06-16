import abc

import numpy as np
from src.pypostEnvironments.dynamicalSystem.DynamicalSystem import DynamicalSystem


class ContinuousTimeDynamicalSystem(DynamicalSystem):

    def __init__(self, rootSampler, dimensions):
        super().__init__(rootSampler, dimensions)

        self.dt = 0.05
        self.linkProperty('dt')

        self.addDataManipulationFunction(self.getLinearizedContinuousTimeDynamics,
                                         ['states', 'actions'],
                                         ['nextStates', 'A', 'B', 'controlNoise'])

    def getControlNoiseStd(self, states, actions, dt=None):
        if dt is None:
            dt = self.dt

        controlNoiseStd = super().getControlNoiseStd(states, actions)
        return controlNoiseStd / np.sqrt(dt)

    def transitionFunction(self, states, actions, *args):
        nextStates, actionNoise = self.transitionFunctionContTime(self.dt, states, actions, *args)
        if self.returnControlNoise:
            return nextStates, actionNoise
        else:
            return nextStates

    def transitionFunctionContTime(self, dt, states, actions, *args):
        actions = np.maximum(self.minRangeAction, np.minimum(actions, self.maxRangeAction))
        actionNoise = self.getControlNoise(states, actions, dt)
        return self.getExpectedNextStateContTime(dt, states, actions + actionNoise, args), actionNoise

    def getExpectedNextState(self, states, actions, *args):
        return self.getExpectedNextStateContTime(self.dt, states, actions, args)


    @abc.abstractmethod
    def getLinearizedContinuousTimeDynamics(self, states, actions, *args):
        return

    @abc.abstractmethod
    def getExpectedNextStateContTime(self, dt, states, actions, *args):
        return

