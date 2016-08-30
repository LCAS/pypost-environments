import abc
from pypost.data.DataManipulator import DataManipulator
class Preprocessor(DataManipulator):

    @abc.abstractmethod
    @DataManipulator.DataMethod(inputArguments=['states'], outputArguments=['flatImages'])
    def preprocessStates(self, states):
        return