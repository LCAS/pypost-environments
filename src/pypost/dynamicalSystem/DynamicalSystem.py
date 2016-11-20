import numpy as np
from pypost.envs import TransitionFunction

class DynamicalSystem(TransitionFunction):

    def __init__(self, dataManager, actionDim, stateDim = None):
        if (not stateDim):
            stateDim = actionDim * 2
        super().__init__(dataManager, stateDim, actionDim)

        self.noiseStd = 1
        self.noiseMode = 0
        self.registerControlNoise = False

        self.linkProperty('noiseStd')
        self.linkProperty('noiseMode')
        self.linkProperty('registerControlNoise')
        self.dt = 0.05
        self.linkProperty('dt')

        if (self.registerControlNoise):
            subdataManager = dataManager.subDataManager
            subdataManager.addDataEntry('actionsNoise', actionDim)


    def getControlNoise(self, states, actions):
        std = self.getControlNoiseStd(states, actions)
        return np.random.normal(loc=0.0, scale=std, size=np.shape(actions))

    @TransitionFunction.DataMethod(inputArguments=['states', 'actions'], outputArguments=['actionsStd'])
    def getControlNoiseStd(self, states, actions):
        if self.noiseMode == 0:
            return self.noiseStd * np.ones(np.shape(actions)) / np.sqrt(self.dt)
        elif self.noiseMode == 1:
            return self.noiseStd * np.abs(actions) / np.sqrt(self.dt)


#    def transitionFunction(self, states, actions):
#        actionNoise = self.getControlNoise(states, actions)
#        new_states = self.getExpectedNextState(states, actions + actionNoise)
#        return new_states

    # Todo Test this, should be in new class
    #def getLinearizedDynamics(self, states, actions, *args):
    #    f_states = np.zeros((self.stateDim, self.stateDim))
    #    f_actions = np.zeros((self.stateDim, self.actionDim))
    #    u_dummy = np.zeros((1, self.actionDim)) #?
#
#        f = self.getExpectedNextState(states, actions, args)
#        assert not np.isnan(f).any() #does this work?
#
#        stepSize = 1e-5
#        # finite differences...
#        for i in range(0, self.stateDim):
#            states_temp = states
#            states_temp[i] = states[i] + stepSize
#            f1 = self.getExpectedNextState(states_temp, actions, args)
#            states_temp[i] = states[i] - stepSize
#            f2 = self.getExpectedNextState(states_temp, actions, args)
#            f_states[:, i] = (f1 - f2) / (2 * stepSize)

#        for i in range(0, self.actionDim):
#            actions_temp = u_dummy
#            actions_temp[i] = u_dummy[i] + stepSize
#            f1 = self.getExpectedNextState(states, actions_temp, args)
#            actions_temp[i] = u_dummy[i] - stepSize
#            f2 = self.getExpectedNextState(states, actions_temp, args)
#            f_actions[:, i] = (f1 - f2) / (2 * stepSize)

#        f = np.transpose(f) - f_states * np.transpose(states) + f_actions * np.transpose(actions)
#        assert not (np.isnan(f).any() or np.isnan(f_states).any() or np.isnan(f_actions).any())

#        if args is not None:
#            controlNoise = self.getControlNoiseStd(states, actions, args)
#        else:
#            controlNoise = None #?


#       return f, f_states, f_actions, controlNoise

#    def getSystemNoiseCovariance(self, states, actions, *args):
#        _, _, B, controlNoise = self.getLinearizedDynamics(states,actions, args)
#        return B * controlNoise * np.transpose(B)

