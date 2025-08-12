from abc import ABC, abstractmethod

class BaseFeatureExtractor(ABC):
    @abstractmethod
    def prepare_tasks(self, **kwargs):
        """Takes a dataset metadata and process task data"""
        pass

    @abstractmethod
    def embed(self, images, **kwargs):
        """Returns encoded outputs (e.g. slots, patches)"""
        pass

    @abstractmethod
    def to(self, device_or_dtype):
        pass

    @abstractmethod
    def device(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass
