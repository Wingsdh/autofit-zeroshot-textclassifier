from abc import ABC, abstractmethod
from typing import List


class TextClassifier(ABC):
    """Base abstract class for text classifier"""

    def __init__(
        self,
    ):
        self._labels = []

    def read_labels(self, label_path):
        labels = []
        with open(label_path, "r", encoding="utf-8") as fd:
            for line in fd:
                labels.append(line.strip())
        self.labels = labels

    @property
    def labels(self):
        """
        support labels, e.g. , [Business & Finance, Entertainment & Music, ...] in topic classes
        """
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value

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
