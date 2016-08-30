import abc

import numpy as np

from pypost.dynamicalSystem import DynamicalSystem


class ContinuousTimeDynamicalSystem(DynamicalSystem):

    def __init__(self, dataManager, dimensions):
        super().__init__(dataManager, dimensions)

        self.dt = 0.05
        self.linkProperty('dt')

    def getControlNoiseStd(self, states, actions, dt=None):
        if dt is None:
            dt = self.dt

        controlNoiseStd = super().getControlNoiseStd(states, actions)
        return controlNoiseStd / np.sqrt(dt)

    def transitionFunction(self, states, actions, *args):
        nextStates, actionNoise = self.transitionFunctionContTime(states, actions, *args)
        if self.returnControlNoise:
            return nextStates, actionNoise
        else:
            return nextStates

    def transitionFunctionContTime(self, states, actions, *args):
        actions = np.maximum(self.minRangeAction, np.minimum(actions, self.maxRangeAction))
        actionNoise = self.getControlNoise(states, actions, self.dt)
        return self.getExpectedNextStateContTime(states, actions + actionNoise, args), actionNoise

    def getExpectedNextState(self, states, actions, *args):
        return self.getExpectedNextStateContTime(states, actions, args)


    @abc.abstractmethod
    @DynamicalSystem.DataMethod(inputArguments=['states', 'actions'],
                                outputArguments=['nextStates', 'A', 'B', 'controlNoise'])
    def getLinearizedContinuousTimeDynamics(self, states, actions, *args):
        return

    @abc.abstractmethod
    @DynamicalSystem.DataMethod(inputArguments=['states', 'actions'], outputArguments=['nextStates'])
    def getExpectedNextStateContTime(self, states, actions, *args):
        return

