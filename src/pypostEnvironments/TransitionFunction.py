from pypost.data.DataManipulator import DataManipulator
from pypost.data.DataManager import DataManager
import abc


# Todo Periodicity? Not supported by Data Manager in Python
class TransitionFunction(DataManipulator):

    def __init__(self, rootSampler, dimState, dimAction):
        DataManipulator.__init__(self, rootSampler.dataManager)
        self.dimState = dimState
        self.dimAction = dimAction

        self.episodeManager = rootSampler.dataManager
        self.stepManager = self.episodeManager.subDataManager
        if self.stepManager is None:
            self.stepManager = self.episodeManager.subDataManager = DataManager('steps')

        self.stepManager.addDataEntry('states', dimState)
        self.stepManager.addDataEntry('nextStates', dimState)
        self.stepManager.addDataEntry('actions', dimAction)

        rootSampler.stepSampler.addElementsForTransition("nextStates", "states")

        self.addDataManipulationFunction(self.transitionFunction, ['states', 'actions'], ['nextStates'])
        self.addDataFunctionAlias('sampleNextState', 'transitionFunction')

        self.addDataManipulationFunction(self.initStateFromContexts, ['contexts'], ['states'])
        self.addDataFunctionAlias('sampleInitState', 'initStateFromContexts')


    def initStateFromContexts(self, contexts):
        return contexts[:, 0: self.dimState]

    @abc.abstractmethod
    def transitionFunction(self, *args):
        return


    def getStateDifference(self, state1, state2):
        stateDiff = (state1 - state2)
        return stateDiff


    def projectStateInPeriod(self, state):
        # todo implement, no periodicity feature in data manager
        raise RuntimeError('Not yet Implemented')


    def initObject(self):
        self.dimState = self.stepManager.getNumDimensions('states')
        self.dimAction = self.stepManager.getNumDimensions('actions')

        self.minRangeState = self.stepManager.getMinRange('states')
        self.maxRangeState = self.stepManager.getMaxRange('states')

        self.minRangeAction = self.stepManager.getMinRange('actions')
        self.maxRangeAction = self.stepManager.getMaxRange('actions')

