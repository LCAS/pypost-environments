import numpy as np
from pypostEnvironments.TransitionFunctionGaussianNoise import TransitionFunctionGaussianNoise


class DynamicalSystem(TransitionFunctionGaussianNoise):

    def __init__(self, rootSampler, dimensions):
        super().__init__(rootSampler, dimensions * 2, dimensions)

        self.noiseStd = 1
        self.noiseMode = 0
        self.returnControlNoise = False

        self.linkProperty('noiseStd')
        self.linkProperty('noiseMode')
        self.linkProperty('returnControlNoise')

        self.addDataManipulationFunction(self.getControlNoiseStd, ['states', 'actions'], ['noise_std'])
        self.addDataManipulationFunction(self.getControlNoise, ['states', 'actions'], ['actionNoise'])

        self.addDataManipulationFunction(self.getTransitionLogProbabilities,
                                         ['states', 'actions', 'actionsNoise'],
                                         ['logProbTrans'])
        self.addDataManipulationFunction(self.getUncontrolledTransitionLogProbabilities,
                                         ['states', 'actions', 'actionsNoise'],
                                         ['logProbTrans'])




    def getControlNoise(self, states, actions, *args):
        std = self.getControlNoiseStd(states, actions)
        return np.random.normal(loc=0.0, scale=std, size=np.shape(actions))

    def getControlNoiseStd(self, states, actions, *args):
        if self.noiseMode == 0:
            return self.noiseStd * np.ones(np.shape(actions))
        elif self.noiseMode == 1:
            return self.noiseStd * np.abs(actions)

    def getTransitionLogProbabilities(self, states, actions, noise):
        std = self.getControlNoiseStd(states, actions)
        std[std < 1e-8] = 1e-8
        noiseNorm = noise / std # check here
        return -0.5 * np.sum(noiseNorm**2, 1)

    def getUncontrolledTransitionLogProbabilities(self, states, actions, noise):
        std = self.getControlNoiseStd(states, actions)
        std[std < 1e-8] = 1e-8
        noiseNorm = (noise + actions) / std
        return -0.5*np.sum(noiseNorm**2, 1)

    def transitionFunction(self, states, actions, *args):
        actionNoise = self.getControlNoise(states, actions, args)

        new_states = self.getExpectedNextState(states, actions + actionNoise, args)
        return new_states, actionNoise

    # Todo Test this
    def getLinearizedDynamics(self, states, actions, *args):
        f_states = np.zeros(self.dimState, self.dimState)
        f_actions = np.zeros(self.dimState, self.dimAction)
        u_dummy = np.zeros(1, self.dimAction) #?

        f = self.getExpectedNextState(states, actions, args)
        assert not np.isnan(f).any() #does this work?

        stepSize = 1e-5
        # finite differences...
        for i in range(0, self.dimState):
            states_temp = states
            states_temp[i] = states[i] + stepSize
            f1 = self.getExpectedNextState(states_temp, actions, args)
            states_temp[i] = states[i] - stepSize
            f2 = self.getExpectedNextState(states_temp, actions, args)
            f_states[:, i] = (f1 - f2) / (2 * stepSize)

        for i in range(0, self.dimAction):
            actions_temp = u_dummy
            actions_temp[i] = u_dummy[i] + stepSize
            f1 = self.getExpectedNextState(states, actions_temp, args)
            actions_temp[i] = u_dummy[i] - stepSize
            f2 = self.getExpectedNextState(states, actions_temp, args)
            f_actions[:, i] = (f1 - f2) / (2 * stepSize)

        f = np.transpose(f) - f_states * np.transpose(states) + f_actions * np.transpose(actions)
        assert not (np.isnan(f).any() or np.isnan(f_states).any() or np.isnan(f_actions).any())

        if args is not None:
            controlNoise = self.getControlNoiseStd(states, actions, args)
        else:
            controlNoise = None #?

        return f, f_states, f_actions, controlNoise

    def getSystemNoiseCovariance(self, states, actions, *args):
        _, _, B, controlNoise = self.getLinearizedDynamics(states,actions, args)
        return B * controlNoise * np.transpose(B)

