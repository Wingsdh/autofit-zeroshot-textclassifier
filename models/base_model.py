from abc import ABC, abstractmethod
from typing import List

from utils.log_utils import logger


class TextClassifier(ABC):
    """Base abstract class for text classifier"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self._labels = []

    def read_labels(self, label_path):
        labels = []
        with open(label_path, "r", encoding="utf-8") as fd:
            for line in fd:
                labels.append(line.strip())
        self.labels = labels
        logger.info(f"Read {len(self.labels)} from {label_path}")
        logger.info(f"Labels: {self.labels}")

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

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
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

    @abstractmethod
    def load(self, model_path: str):
        """load.

        Parameters
        ----------
        model_path : str
            model_path
        """
        raise NotImplementedError
