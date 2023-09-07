import os
import sys
import glob
from typing import Any
from rackio_AI import RackioAI

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
    def _load_data(self) -> list:
        """Loads the data in pkl format
        """
        path = os.path.join(*self.configs['DATA_PATH']['path'])
        filename = self.configs['DATA_PATH']['filename']
        self.data = list()
        files = list()
        if filename:

            files.extend([os.path.join(path, filename)])

        else:

            path = os.path.join(*self.configs['DATA_PATH']['path']) + os.path.sep + '*.pkl'
            files = glob.glob(path)

        for file in files:

            self.data.extend(RackioAI.load(file))
        
        return self.data
    
    def run(self):
        self._load_data()
        
    def __call__(self) -> Any:
        self.run()