import os
from typing import List

import numpy as np
from gensim import corpora, downloader, models

from models import base_model
from utils.log_utils import logger
from utils.token_utils import text_to_wordlist


def gensim_load_vec(model_name="word2vec-google-news-300") -> models.KeyedVectors:
    logger.info(f"Loading vec {model_name}")
    gensim_emb = downloader.load(model_name)
    logger.info("Done")
    return gensim_emb


class MeanWord2vecClassifier(base_model.TextClassifier, base_model.SelfTrainable):
    """docstring for MeanClassifier."""

    def __init__(self, model_name="word2vec-google-news-300", *args, **kwarg):
        super(MeanWord2vecClassifier, self).__init__(*args, **kwarg)
        self.model_name = model_name

    def classify(self, text: str) -> str:
        tokens = [
            t
            for t in text_to_wordlist(text, remove_stopwords=True, stem_words=True)
            if t in self.emb.key_to_index
        ]
        class_emb = [self.emb.get_vector(label, norm=True) for label in self.labels]
        text_emb = self.emb.get_mean_vector(keys=tokens, post_normalize=True)

        ori_scores = np.dot(class_emb, text_emb.T)
        return self.labels[np.argmax(ori_scores)]

    def _fit_dictionary(self, texts, token_set):
        tokenized_texts = [
            [
                t
                for t in text_to_wordlist(text, remove_stopwords=True, stem_words=True)
                if t in token_set
            ]
            for text in texts
        ]
        dictionary = corpora.Dictionary.from_documents(tokenized_texts)
        return dictionary

    def fit(self, texts: List[str]):
        """fit.

        Parameters
        ----------
        texts : List[str]
            texts
        """
        # Load all emb
        gensim_emb = gensim_load_vec(self.model_name)

        # fit vocab
        dictionary = self._fit_dictionary(texts, gensim_emb.key_to_index)

        # prune emb
        fitted_gensim_emb = models.KeyedVectors(
            gensim_emb.vector_size, count=len(dictionary)
        )
        for label in self.labels:
            fitted_gensim_emb.add_vector(label, gensim_emb.get_vector(label))

        for token, index in dictionary.token2id.items():
            fitted_gensim_emb.add_vector(token, gensim_emb.get_vector(token))

        self.emb = fitted_gensim_emb

    @staticmethod
    def get_emb_path(model_path):
        return os.path.join(model_path, "word2vec.bin")

    def save(self, model_path: str):
        """save.

        Parameters
        ----------
        model_path : str
            model_path
        """
        if not os.path.exists(model_path):
            logger.info(f"No model_path: {model_path}, make dir")
            os.makedirs(model_path)

        emb_path = self.get_emb_path(model_path)
        logger.info(f"Saving emb into {emb_path}")
        self.emb.save(emb_path)
        logger.info("Saved")

    def load(self, model_path: str):
        """load.

        Parameters
        ----------
        model_path : str
            model_path
        """
        emb_path = self.get_emb_path(model_path)
        if not os.path.exists(emb_path):
            return False
        self.emb = models.KeyedVectors.load(emb_path)
        return True


class TfidfClassifier(base_model.TextClassifier, base_model.SelfTrainable):
    """docstring for TfidfClassifier."""

    def __init__(
        self,
    ):
        super(TfidfClassifier, self).__init__()

    def classify(self, text: str) -> str:
        pass
