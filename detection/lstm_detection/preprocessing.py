import os
import sys
from typing import Any

# This code line is to avoid import relative error
sys.path.append("..")

from utils import parse_config, convert_list2tuple, report_done
from feature_extraction import FeatureExtraction

class DetectionPreprocessPipeline:
    """This class preprocess the data for training LSTM classification model for detection.
    """
    def __init__(self, configs: str) -> None:
        self.configs = parse_config(configs)

    @report_done
    def _load_data(self):
        pass
    
    def run(self):
        self._load_data()

    def __call__(self) -> Any:
        self.run()