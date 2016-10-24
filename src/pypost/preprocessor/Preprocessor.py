import abc
from pypost.data import DataManipulator
class Preprocessor(DataManipulator):

    @abc.abstractmethod
    @DataManipulator.DataMethod(inputArguments=['states'], outputArguments=['flatImages'])
    def preprocessStates(self, states):
        return