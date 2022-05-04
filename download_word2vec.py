import fire
import gensim
from gensim import downloader

from utils.log_utils import logger
import nltk


def download(model_name):
    logger.info
    nltk.download('stopwords')
    logger.info(f"Downloading {model_name}")
    try:
        downloader.load(model_name)
    except ValueError:
        valid_models = list(gensim.downloader.info()["models"].keys())

        logger.error(f"Invalid model_name {model_name}, must be one of {valid_models}")

    logger.info(f"Finish download {model_name}")


if __name__ == "__main__":
    fire.Fire(download)
