import abc
class Preprocessor():

    @abc.abstractmethod
    def preprocessData(self, data, *args):
        return