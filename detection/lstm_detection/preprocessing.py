import os
import sys

# # This code line is to avoid import relative error
sys.path.append("...")

from utils import parse_config

class DetectionPreprocessPipeline:
    """This class preprocess the data for training LSTM classification model for detection.
    """
    def __init__(self, configs: str) -> None:
        self.configs = parse_config(configs)

