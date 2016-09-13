from pypost.data import DataManipulator
import numpy as np

# Dummy Reward and Action Function for testing
class DummyActionAndReward(DataManipulator):

    def __init__(self, dataManager, dimAction, generateActions = False):
        super().__init__(dataManager)
        self.dimAction = dimAction
        self.generateActions = generateActions
        # magic number
        self.maxTorque = 30

        dataManager.addDataEntry('rewards', 1)

    @DataManipulator.DataMethod(inputArguments=[], outputArguments=['rewards'], takesNumElements=True)
    def sampleReward(self, numElem):
        return np.zeros((numElem, 1))

    @DataManipulator.DataMethod(inputArguments=[], outputArguments=['actions'], takesNumElements=True)
    def sampleAction(self, numElem):
        if self.generateActions:
            return np.random.uniform(-self.maxTorque, self.maxTorque, [numElem, self.dimAction])
        else:
            return np.zeros([numElem, self.dimAction])

