import fire

from models import word2vec_based
from utils import data_utils
from utils.log_utils import logger


def main(
    model_path, label_path, text_path, annotation_path, w2v="word2vec-google-news-300"
):
    cls = word2vec_based.MeanWord2vecClassifier()
    cls.read_labels(label_path)
    data = data_utils.load_data(text_path)
    annotations = data_utils.load_data(annotation_path)
    loaded = cls.load(model_path)
    if not loaded:
        cls.fit(texts=data)
        cls.save(model_path)

    right = 0
    for t, a in zip(data, annotations):
        rst = cls.classify(t)
        if rst == cls.labels[int(a)]:
            right += 1
    logger.info(f"Acc: {right} / {len(data)} = {right/len(data)}")


if __name__ == "__main__":
    fire.Fire(main)
