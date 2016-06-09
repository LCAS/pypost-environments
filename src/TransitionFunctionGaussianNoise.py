from src.TransitionFunction import TransitionFunction
import abc

class TransitionFunctionGaussianNoise(TransitionFunction):

    def __init__(self, rootSampler, dimState, dimAction):
        super().__init__(rootSampler, dimState, dimAction)

        self.addDataManipulationFunction(self.getExpectedNextState, ['states', 'actions'], ['nextStates'])
        self.addDataManipulationFunction(self.getSystemNoiseCovariance, ['states', 'actions'], ['systemNoise'])


    @abc.abstractmethod
    def getExpectedNextState(self, *args):
        return

    @abc.abstractmethod
    def getSystemNoiseCovariance(self, *args):
        return



