from abc import ABC, abstractmethod

class Reducer(ABC):
    @abstractmethod
    def reduce(self, data):
        pass
    