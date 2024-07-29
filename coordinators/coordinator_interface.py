from abc import ABC, abstractmethod

class CoordinatorInterface(ABC):
    @abstractmethod
    def coordinate(self, *args, **kwargs):
        pass