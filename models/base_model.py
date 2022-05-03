from abc import ABC, abstractmethod
from typing import List


class TextClassifier(ABC):
    """Base abstract class for text classifier"""

    def labels(self) -> List[str]:
        """
        support labels, e.g. , [Business & Finance, Entertainment & Music, ...] in topic classes
        """
        raise NotImplementedError

    @abstractmethod
    def classify(self, text: str) -> str:
        """classify.

        Parameters
        ----------
        text :
            The text to be classified

        Returns
        -------
        str : label which is defined in self.labels
        """
        raise NotImplementedError


class SelfTrainable(ABC):
    """The model can be trained by a unlabeled text corpus"""

    def fit(self, texts: List[str]):
        """fit.

        Parameters
        ----------
        texts : List[str]
            texts
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, model_path: str):
        """save.

        Parameters
        ----------
        model_path : str
            model_path
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, model_path: str):
        """load.

        Parameters
        ----------
        model_path : str
            model_path
        """
        raise NotImplementedError
