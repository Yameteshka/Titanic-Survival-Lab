import logging

from code.datasets.preprocess import clean_and_split_data
from code.models.evaluate import evaluate_and_log
from code.models.train import train_model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    clean_and_split_data()
    train_model()
    evaluate_and_log()
